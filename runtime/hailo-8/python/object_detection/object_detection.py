#!/usr/bin/env python3
import argparse
import os
import sys
from loguru import logger
import queue
import threading
from functools import partial
from types import SimpleNamespace
import time
from pathlib import Path
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.tracker.byte_tracker import BYTETracker
from common.hailo_inference import HailoAsyncInference
from common.toolbox import init_input_source, get_labels, load_json_file, preprocess, visualize
from object_detection_post_process import inference_result_handler

frame_counter = [0]  # Using a mutable list to share counter


def parse_args() -> argparse.Namespace:
    """
    Initialize argument parser for the script.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Detection Example")

    parser.add_argument("-n", "--net", help="Path for the network in HEF format.",
                        default="yolov8n.hef")
    parser.add_argument("-i", "--input", default="bus.jpg",
                        help="Path to the input - either an image or a folder of images.")
    parser.add_argument("-b", "--batch_size", default=1, type=int, required=False,
                        help="Number of images in one batch")
    parser.add_argument("-l", "--labels",
                        default=str(Path(__file__).parent.parent / "common" / "coco.txt"),
                        help="Path to a text file containing labels. If no labels file is provided, coco2017 will be used.")
    parser.add_argument("-s", "--save_stream_output", action="store_true",
                        help="Save the output of the inference from a stream.")
    parser.add_argument("-o", "--output-dir", help="Directory to save the results.",
                        default=None)
    parser.add_argument("-r", "--resolution", choices=["sd", "hd", "fhd"], default="sd",
                        help="Choose input resolution: 'sd' (640x480), 'hd' (1280x720), or 'fhd' (1920x1080). Default is 'sd'.")
    parser.add_argument("--track", action="store_true",
                        help="Enable object tracking across frames.")
    parser.add_argument("--show-fps", action="store_true",
                        help="Enable FPS performance measurement.")

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.net):
        raise FileNotFoundError(f"Network file not found: {args.model}")
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Labels file not found: {args.labels}")

    if args.output_dir is None:
        args.output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def infer(net, input, batch_size, labels, output_dir,
          save_stream_output=False, resolution="sd",
          enable_tracking=False, show_fps=False) -> None:
    """
    Initialize queues, HailoAsyncInference instance, and run the inference.
    """
    labels = get_labels(labels)
    config_data = load_json_file("config.json")

    # Initialize input source from string: "camera", video file, or image folder.
    cap, images = init_input_source(input, batch_size, resolution)
    tracker = None

    if enable_tracking:
        # load tracker config from config_data
        tracker_config = config_data.get("visualization_params", {}).get("tracker", {})
        tracker = BYTETracker(SimpleNamespace(**tracker_config))

    input_queue = queue.Queue()
    output_queue = queue.Queue()

    post_process_callback_fn = partial(
        inference_result_handler, labels=labels,
        config_data=config_data, tracker=tracker
    )
    inference_callback_fn = partial(
        inference_callback, output_queue=output_queue
    )

    hailo_inference = HailoAsyncInference(
        net, input_queue, inference_callback_fn,
        batch_size, send_original_frame=True
    )
    height, width, _ = hailo_inference.get_input_shape()

    preprocess_thread = threading.Thread(
        target=preprocess, args=(images, cap, batch_size, input_queue, width, height)
    )
    postprocess_thread = threading.Thread(
        target=visualize, args=(output_queue, cap, save_stream_output,
                                output_dir, post_process_callback_fn, frame_counter)
    )

    if show_fps:
        start_time = time.time()

    preprocess_thread.start()
    postprocess_thread.start()
    hailo_inference.run()

    preprocess_thread.join()
    output_queue.put(None)  # Signal process thread to exit
    postprocess_thread.join()

    logger.info('Inference was successful!')

    if show_fps:
        end_time = time.time()
        fps = frame_counter[0] / (end_time - start_time)
        logger.debug(f"Processed {frame_counter[0]} frames at {fps:.2f} FPS")


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


def main() -> None:
    """
    Main function to run the script.
    """
    args = parse_args()
    infer(args.net, args.input, args.batch_size, args.labels,
          args.output_dir, args.save_stream_output, args.resolution,
          args.track, args.show_fps)


if __name__ == "__main__":
    main()