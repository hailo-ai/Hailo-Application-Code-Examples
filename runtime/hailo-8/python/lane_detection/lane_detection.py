#!/usr/bin/env python3
import multiprocessing as mp
import argparse
import sys
import os
from multiprocessing import Process

import numpy as np
from loguru import logger
import cv2

from lane_detection_utils import (UFLDProcessing,
                                  check_process_errors,
                                  output_data_type2dict,
                                  compute_scaled_radius)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference

def parser_init():
    """
    Initialize and configure the argument parser for this script.
    
    Returns:
        argparse.ArgumentParser: An instance of ArgumentParser.
    """
    parser = argparse.ArgumentParser(description="UFLD_v2 inference")

    parser.add_argument(
        "-n",
        "--net",
        help="Path of model file in HEF format.",
        default="ufld_v2.hef"
    )

    parser.add_argument(
        "-i",
        "--input_video",
        default="input_video.mp4",
        help="Path of the video to perform inference on.",
    )

    parser.add_argument(
        "-o",
        "--output_video",
        default="output_video.mp4",
        help="Path of the output video.",
    )

    return parser


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
        input_queue.put((np.array([frame]),
                         np.array([resized_frame])))
        success, frame = vidcap.read()

    input_queue.put(None)  # Sentinel value to signal the end of processing


def postprocess_output(output_queue: mp.Queue,
                       output_video_path: str,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
                       ufld_processing: UFLDProcessing) -> None:
    """
    Post-process inference results, draw lane detections, and write output to a video.

    Args:
        output_queue (mp.Queue): Queue for output results.
        output_video_path (str): Path to the output video file.
        ufld_processing (UFLDProcessing): Lane detection post-processing class.
    """
    # Import tqdm here to avoid issues with multiprocessing
    from tqdm import tqdm

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width, height = ufld_processing.get_original_frame_size()
    output_video = cv2.VideoWriter(output_video_path, fourcc, 20,
                                   (width, height))

    # Compute the scaled radius for the lane detection points
    radius = compute_scaled_radius(width, height)

    pbar = tqdm(total=total_frames, desc="Processing frames")

    while True:
        result = output_queue.get()
        if result is None:
            break  # Exit when the sentinel value is received
        original_frame, inference_output = result
        lanes = ufld_processing.get_coordinates(np.array([inference_output]))

        for lane in lanes:
            for coord in lane:
                cv2.circle(original_frame, coord, radius, (0, 255, 0), -1)
        output_video.write(original_frame.astype('uint8'))
        pbar.update(1)

    pbar.close()
    output_video.release()

def infer(
    video_path: str,
    net_path: str,
    batch_size: int,
    output_video_path: str,
    ufld_processing: UFLDProcessing
) -> None:
    """
    Run lane detection inference using HailoAsyncInference and manage the video processing pipeline.

    Args:
        video_path (str): Path to the input video.
        net_path (str): Path to the HEF model file.
        batch_size (int): Number of frames per batch.
        output_video_path (str): Path to save the output video.
        ufld_processing (UFLDProcessing): Lane detection processing class.
    """

    input_queue = mp.Queue()
    output_queue = mp.Queue()

    output_dict = output_data_type2dict(net_path, "FLOAT32")

    hailo_inference = HailoAsyncInference(net_path,input_queue,
                                          output_queue, batch_size,
                                          output_type=output_dict,
                                          send_original_frame=True)


    preprocessed_frame_height, preprocessed_frame_width, _ = hailo_inference.get_input_shape()
    preprocess = Process(
        target=preprocess_input,
        args=(video_path,
              input_queue,
              preprocessed_frame_width,
              preprocessed_frame_height,
              ufld_processing)
    )
    postprocess = Process(
        target=postprocess_output,
        args=(output_queue, output_video_path, ufld_processing)
    )

    preprocess.start()
    postprocess.start()

    try:
        hailo_inference.run()
        preprocess.join()

        # Signal to the postprocess to stop
        output_queue.put(None)
        postprocess.join()

        check_process_errors(preprocess, postprocess)
        logger.info(f"Inference was successful! Results saved in {output_video_path}")

    except Exception as e:
        logger.error(f"Inference error: {e}")
        input_queue.close()
        output_queue.close()
        preprocess.terminate()
        postprocess.terminate()
        os._exit(1)


if __name__ == "__main__":

    # Parse command-line arguments
    args = parser_init().parse_args()
    try:
        original_frame_width,original_frame_height, total_frames= get_video_info(args.input_video)
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

    infer(
        args.input_video,
        args.net,
        batch_size=1,
        output_video_path=args.output_video,
        ufld_processing=ufld_processing
    )
