#!/usr/bin/env python3
import numpy as np
import cv2
from pathlib import Path
import os
from loguru import logger
import argparse
import sys
from typing import List
import threading
import queue
from super_resolution_utils import SrganUtils, Espcnx4Utils, inference_result_handler
from functools import partial

# Add the parent directory to the system path to access utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.hailo_inference import HailoInfer
from common.toolbox import  init_input_source, visualize, preprocess, FrameRateTracker

def parse_args() -> argparse.Namespace:
    """
    Initialize argument parser for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Super Resolution Example")
    parser.add_argument(
        "-n", "--net", 
        help="Path for the network in HEF format.",
        default="real_esrgan_x2.hef"
    )
    parser.add_argument(
        "-i", "--input", default="ocr_img1.png",
        help="Path to the input - either an image or a folder of images."
    )
    parser.add_argument(
        "-b", "--batch_size", default=1, type=int, required=False,
        help="Number of images in one batch"
    )
    parser.add_argument(
        "-s", "--save_stream_output", action="store_true",
        help="Save the output of the inference from a stream."
    )
    parser.add_argument(
        "-o", "--output-dir", help="Directory to save the results.",
        default=None
    )
    parser.add_argument(
        "-r", "--resolution",
        choices=["sd", "hd", "fhd"],
        default="sd",
        help="Camera only: Choose input resolution: 'sd' (640x480), 'hd' (1280x720), or 'fhd' (1920x1080). Default is 'sd'."
    )

    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Enable FPS measurement and display."
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.net):
        raise FileNotFoundError(f"Network file not found: {args.net}")

    if args.output_dir is None:
        args.output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(args.output_dir, exist_ok=True)


    return args



def inference_callback(
        completion_info,
        bindings_list: list,
        input_batch: list,
        output_queue: queue.Queue
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
    input: str,
    batch_size: int,
    output_dir: str,
    save_stream_output: bool,
    resolution: str,
    show_fps: bool
) -> None:
    """
    Initialize queues, create HailoAsyncInference instance, and run the inference pipeline.

    Args:
        net_path (str): Path to the HEF model file.
        input (str): Input source path (image directory, video file, or camera).
        batch_size (int): Number of frames to process per batch.
        output_dir (str): Directory path to save output visualizations.
        save_stream_output (bool): Whether to save the processed stream to video/images.
        resolution (str): Input resolution, e.g., 'sd', 'hd', or 'fhd'.
        show_fps (bool): Whether to print/log FPS (frames per second) information during execution.

    Returns:
        None
    """

    utils = None
    # Initialize input source from string: "camera", video file, or image folder.
    cap, images = init_input_source(input, batch_size, resolution)

    input_queue = queue.Queue()
    output_queue = queue.Queue()


    fps_tracker = None
    if show_fps:
        fps_tracker = FrameRateTracker()

    if 'espcn' in net_path:
        utils = Espcnx4Utils()
        hailo_inference = HailoInfer(net_path, batch_size, input_type="FLOAT32", output_type="FLOAT32")
    else:
        utils = SrganUtils()
        hailo_inference = HailoInfer(net_path, batch_size)
    
    height, width, _ = hailo_inference.get_input_shape()

    post_process_callback_fn = partial(
        inference_result_handler,model_height=height, model_width=width
    )

    preprocess_thread = threading.Thread(
        target=preprocess, args=(images, cap, batch_size, input_queue, width, height)
    )
    postprocess_thread = threading.Thread(
        target=visualize, args=(output_queue, cap, save_stream_output,
                                output_dir, post_process_callback_fn, fps_tracker, True)
    )
    infer_thread = threading.Thread(
        target=infer, args=(hailo_inference, input_queue, output_queue)
    )

    preprocess_thread.start()
    postprocess_thread.start()
    infer_thread.start()


    if show_fps:
        fps_tracker.start()

    preprocess_thread.join()
    infer_thread.join()
    output_queue.put(None)  # Signal process thread to exit
    postprocess_thread.join()

    if show_fps:
        logger.debug(fps_tracker.frame_rate_summary())


    logger.info('Inference was successful!')



def main() -> None:
    """
    Main function to run the script.
    """
    # Parse command line arguments
    args = parse_args()

    # Start the inference
    run_inference_pipeline(args.net, args.input, args.batch_size, args.output_dir,
          args.save_stream_output, args.resolution, args.show_fps)
if __name__ == "__main__":
    main()
