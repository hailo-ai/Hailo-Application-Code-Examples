#!/usr/bin/env python3
import os
import sys
import argparse
import multiprocessing as mp
from queue import Queue
from loguru import logger
from functools import partial
import numpy as np
import threading
from pose_estimation_utils import PoseEstPostProcessing
from pathlib import Path

# Add the parent directory to the system path to access utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.hailo_inference import HailoInfer
from common.toolbox import (
    init_input_source,
    preprocess,
    visualize,
    FrameRateTracker,
    resolve_net_arg,
    resolve_input_arg,
    resolve_output_resolution_arg,
    list_networks,
    list_inputs
)

APP_NAME = Path(__file__).stem

def parse_args() -> argparse.Namespace:
    """
    Initialize argument parser for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Running a Hailo inference with actual images using Hailo API and OpenCV",
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
            "  - Special:    'camera'\n"
            "  - Named resource (without extension), e.g. 'bus'.\n"
            "    If a named resource is used, it will be downloaded automatically\n"
            "    if not already available. Use --list-inputs to see the options."
        )
    )
    parser.add_argument(
        "-b", "--batch_size",
        default=1,
        type=int,
        required=False,
        help="Number of images in one batch. Defaults to 1"
    )
    parser.add_argument(
        "-cn", "--class_num",
        help="The number of classes the model is trained on. Defaults to 1",
        default=1
    )
    parser.add_argument(
        "-s", "--save_stream_output", action="store_true",
        help="Save the output of the inference from a stream."
    )
    parser.add_argument(
        "-o", "--output-dir", help="Directory to save the results.",
        default=None
    )
    display_group = parser.add_mutually_exclusive_group(required=False)
    display_group.add_argument(
        "-cr","--camera-resolution",
        type=str,
        choices=["sd", "hd", "fhd"],
        help="(Camera only) Input resolution: 'sd' (640x480), 'hd' (1280x720), or 'fhd' (1920x1080)."
    )
    display_group.add_argument(
        "-or","--output-resolution",
        nargs="+",
        type=str,
        help=(
            "(Camera only) Output resolution. Use: 'sd', 'hd', 'fhd', "
            "or custom size like '--output-resolution 1920 1080'."
        )
    )
    parser.add_argument(
        "-f", "--framerate",
        type=float,
        default=30.0,
        help=("[Camera only] Override the camera input framerate.\n"
            "Example: -f 10.0")
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Enable FPS measurement and display."
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
    args.output_resolution = resolve_output_resolution_arg(args.output_resolution)

    if args.output_dir is None:
        args.output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(args.output_dir, exist_ok=True)

    return args


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
    net_path: str,
    input :str,
    batch_size: int,
    class_num: int,
    output_dir: str,
    camera_resolution: str,
    output_resolution: str,
    framerate: float,
    save_stream_output :bool,
    show_fps: bool
) -> None:
    """
    Run the inference pipeline using HailoInfer.

    Args:
        net_path (str): Path to the HEF model file.
        input (str): Path to the input source (image, video, folder, or camera).
        batch_size (int): Number of frames to process per batch.
        class_num (int): Number of output classes expected by the model.
        output_dir (str): Directory where processed output will be saved.
        save_stream_output (bool): If True, saves the output stream as a video file.
        resolution (str): Camera only, resolution of the input source (e.g., "1280x720").
        show_fps (bool): If True, display real-time FPS on the output.

    Returns:
        None
    """
    input_queue = Queue()
    output_queue = Queue()


    pose_post_processing = PoseEstPostProcessing(
        max_detections=300,
        score_threshold=0.001,
        nms_iou_thresh=0.7,
        regression_length=15,
        strides=[8, 16, 32]
    )

    # Initialize input source from string: "camera", video file, or image folder.
    cap, images = init_input_source(input, batch_size, camera_resolution)

    fps_tracker = None
    if show_fps:
        fps_tracker = FrameRateTracker()

    hailo_inference = HailoInfer(
        net_path, batch_size, output_type="FLOAT32")
    height, width, _ = hailo_inference.get_input_shape()

    post_process_callback_fn = partial(
        pose_post_processing.inference_result_handler,
        model_height=height,
        model_width=width,
        class_num = class_num
    )

    preprocess_thread = threading.Thread(
        target=preprocess,
        args=(images, cap, framerate, batch_size, input_queue, width, height)
    )

    postprocess_thread = threading.Thread(
        target=visualize,
        args=(output_queue, cap, save_stream_output,
            output_dir, post_process_callback_fn, fps_tracker, output_resolution, framerate)
        )

    infer_thread = threading.Thread(
        target=infer,
        args=(hailo_inference, input_queue, output_queue)
    )

    infer_thread.start()
    preprocess_thread.start()
    postprocess_thread.start()

    if show_fps:
        fps_tracker.start()
    infer_thread.join()
    preprocess_thread.join()
    output_queue.put(None)     # To signal processing process to exit
    postprocess_thread.join()

    if show_fps:
        logger.debug(fps_tracker.frame_rate_summary())

    logger.success("Inference was successful!")
    if save_stream_output or input.lower() != "camera":
        logger.success(f'Results have been saved in {output_dir}')


def main() -> None:
    args = parse_args()
    run_inference_pipeline(
        args.net, args.input,
        int(args.batch_size),
        int(args.class_num),
        args.output_dir,
        args.camera_resolution,
        args.output_resolution,
        args.framerate,
        args.save_stream_output,
        args.show_fps
    )


if __name__ == "__main__":
    main()
