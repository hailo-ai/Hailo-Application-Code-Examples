#!/usr/bin/env python3
import multiprocessing as mp
import argparse
import sys
import os
from multiprocessing import Process
from functools import partial
from pathlib import Path
import numpy as np
from loguru import logger
import cv2
import threading

from lane_detection_utils import (UFLDProcessing,
                                  check_process_errors,
                                  compute_scaled_radius)

# Add the parent directory to the system path to access utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.hailo_inference import HailoInfer
from common.toolbox import (
    resolve_net_arg,
    resolve_input_arg,
    list_networks,
    list_inputs
)

APP_NAME = Path(__file__).stem


def parser_init():
    """
    Initialize and configure the argument parser for this script.
    
    Returns:
        argparse.ArgumentParser: An instance of ArgumentParser.
    """
    parser = argparse.ArgumentParser(description="UFLD_v2 inference",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-n", "--net",
        type=str,
        help=(
            "- A local HEF file path\n"
            "    → uses the specified HEF directly.\n"
            "- A model name (e.g., yolov8n)\n"
            "    → automatically downloads & resolves the correct HEF for your device.\n"
            "      Use --list-nets to see the available nets."
        )    
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=None,
        help=(
            "Input source. Examples:\n"
            "  - Local path: 'bus.jpg', 'video.mp4', 'images_dir/'\n"
            "  - Named resource (without extension), e.g. 'bus'.\n"
            "    If a named resource is used, it will be downloaded automatically\n"
            "    if not already available. Use --list-inputs to see the options."
        )
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=None,
        help="Path of the output video.",
    )
    parser.add_argument(
        "--list-nets",
        action="store_true",
        help="List supported nets for this app and exit"
    )
    parser.add_argument(
        "--list-inputs",
        action="store_true",
        help="List predefined sample inputs for this app and exit."
    )
    args = parser.parse_args()

    # Handle --list-nets and exit
    if args.list_nets:
        list_networks(APP_NAME)
        sys.exit(0)

    # Handle --list-inputs and exit
    if args.list_inputs:
        list_inputs(APP_NAME)
        sys.exit(0)

    args.net = resolve_net_arg(APP_NAME, args.net, ".")
    args.input = resolve_input_arg(APP_NAME, args.input)
    
    if args.output_dir is None:
        args.output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args


def get_video_info(video_path):
    """
    Get the dimensions (width and height).

    Args:
        video_path (str): Path to the input video file.

    Returns:
        Tuple[int, int]: A tuple containing frame width and frame height.

    Raises:
        ValueError: If the video file cannot be opened.
    """
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        vidcap.release()
        logger.error(f"Cannot open video file {video_path}")
        raise ValueError(f"Cannot open video file {video_path}")
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    vidcap.release()
    return frame_width, frame_height, frame_count


def preprocess_input(video_path: str,
                     input_queue: mp.Queue, width: int, height: int,
                     ufld_processing: UFLDProcessing) -> None:
    """
    Read video frames, preprocess them, and put them into the input queue for inference.

    Args:
        video_path (str): Path to the input video.
        input_queue (mp.Queue): Queue for input frames.
        width (int): Input frame width for resizing.
        height (int): Input frame height for resizing.
        ufld_processing (UFLDProcessing): Lane detection preprocessing class.
    """
    vidcap = cv2.VideoCapture(video_path)
    success, frame = vidcap.read()

    while success:
        resized_frame = ufld_processing.resize(frame, height, width)
        input_queue.put(([frame], [resized_frame]))


        success, frame = vidcap.read()

    input_queue.put(None)  # Sentinel value to signal the end of processing


def postprocess_output(output_queue: mp.Queue,
                       output_dir: str,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
                       ufld_processing: UFLDProcessing) -> None:
    """
    Post-process inference results, draw lane detections, and write output to a video.

    Args:
        output_queue (mp.Queue): Queue for output results.
        output_dir (str): Path to the output video file.
        ufld_processing (UFLDProcessing): Lane detection post-processing class.
    """
    # Import tqdm here to avoid issues with multiprocessing
    from tqdm import tqdm

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width, height = ufld_processing.get_original_frame_size()

    out_path = os.path.join(output_dir, "output.avi")
    output_video = cv2.VideoWriter(out_path, fourcc, 20, (width, height))

    # Compute the scaled radius for the lane detection points
    radius = compute_scaled_radius(width, height)

    pbar = tqdm(total=total_frames, desc="Processing frames")

    while True:
        result = output_queue.get()
        if result is None:
            break  # Exit when the sentinel value is received
        original_frame, inference_output = result
        slices = list(inference_output.values())
        output_tensor = np.concatenate(slices, axis=1)  # Shape: (1, total_features)
        lanes = ufld_processing.get_coordinates(output_tensor)


        for lane in lanes:
            for coord in lane:
                cv2.circle(original_frame, coord, radius, (0, 255, 0), -1)
        output_video.write(original_frame.astype('uint8'))
        pbar.update(1)

    pbar.close()
    output_video.release()



def inference_callback(
        completion_info,
        bindings_list: list,
        input_batch: list,
        output_queue: mp.Queue
) -> None:
    """
    infernce callback to handle inference results and push them to a queue.

    Args:
        completion_info: Hailo inference completion info.
        bindings_list (list): Output bindings for each inference.
        input_batch (list): Original input frames.
        output_queue (queue.Queue): Queue to push output results to.
    """
    if completion_info.exception:
        logger.error(f'Inference error: {completion_info.exception}')
    else:
        for i, bindings in enumerate(bindings_list):
            if len(bindings._output_names) == 1:
                result = bindings.output().get_buffer()
            else:
                result = {
                    name: np.expand_dims(
                        bindings.output(name).get_buffer(), axis=0
                    )
                    for name in bindings._output_names
                }
            output_queue.put((input_batch[i], result))



def infer(hailo_inference, input_queue, output_queue):
    """
    Main inference loop that pulls data from the input queue, runs asynchronous
    inference, and pushes results to the output queue.

    Each item in the input queue is expected to be a tuple:
        (input_batch, preprocessed_batch)
        - input_batch: Original frames (used for visualization or tracking)
        - preprocessed_batch: Model-ready frames (e.g., resized, normalized)

    Args:
        hailo_inference (HailoInfer): The inference engine to run model predictions.
        input_queue (queue.Queue): Provides (input_batch, preprocessed_batch) tuples.
        output_queue (queue.Queue): Collects (input_frame, result) tuples for visualization.

    Returns:
        None
    """
    while True:
        next_batch = input_queue.get()
        if not next_batch:
            break  # Stop signal received

        input_batch, preprocessed_batch = next_batch

        # Prepare the callback for handling the inference result
        inference_callback_fn = partial(
            inference_callback,
            input_batch=input_batch,
            output_queue=output_queue
        )

        # Run async inference
        hailo_inference.run(preprocessed_batch, inference_callback_fn)

    # Release resources and context
    hailo_inference.close()



def run_inference_pipeline(
    video_path: str,
    net_path: str,
    batch_size: int,
    output_dir: str,
    ufld_processing: UFLDProcessing
) -> None:
    """
    Run lane detection inference using HailoAsyncInference and manage the video processing pipeline.

    Args:
        video_path (str): Path to the input video.
        net_path (str): Path to the HEF model file.
        batch_size (int): Number of frames per batch.
        output_dir (str): Path to save the output video.
        ufld_processing (UFLDProcessing): Lane detection processing class.
    """

    input_queue = mp.Queue()
    output_queue = mp.Queue()
    hailo_inference = HailoInfer(net_path, batch_size, output_type="FLOAT32")


    preprocessed_frame_height, preprocessed_frame_width, _ = hailo_inference.get_input_shape()
    preprocess_thread = threading.Thread(
        target=preprocess_input,
        args=(video_path,
              input_queue,
              preprocessed_frame_width,
              preprocessed_frame_height,
              ufld_processing)
    )
    postprocess_thread = threading.Thread(
        target=postprocess_output,
        args=(output_queue, output_dir, ufld_processing)
    )

    infer_thread = threading.Thread(
        target=infer, args=(hailo_inference, input_queue, output_queue)
    )

    preprocess_thread.start()
    postprocess_thread.start()
    infer_thread.start()

    infer_thread.join()
    preprocess_thread.join()

    # Signal to the postprocess to stop
    output_queue.put(None)
    postprocess_thread.join()

    logger.info(f"Inference was successful! Results saved in {output_dir}")



if __name__ == "__main__":

    # Parse command-line arguments
    args = parser_init()
    try:
        original_frame_width, original_frame_height, total_frames= get_video_info(args.input)
    except ValueError as e:
        logger.error(e)

    ufld_processing = UFLDProcessing(num_cell_row=100,
                                     num_cell_col=100,
                                     num_row=56,
                                     num_col=41,
                                     num_lanes=4,
                                     crop_ratio=0.8,
                                     original_frame_width = original_frame_width,
                                     original_frame_height = original_frame_height,
                                     total_frames = total_frames)

    run_inference_pipeline(
        args.input,
        args.net,
        batch_size=1,
        output_dir=args.output_dir,
        ufld_processing=ufld_processing
    )
