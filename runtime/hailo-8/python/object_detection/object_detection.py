#!/usr/bin/env python3
import argparse
import os
import sys
from loguru import logger
import queue
import threading
from functools import partial
from types import SimpleNamespace
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.tracker.byte_tracker import BYTETracker
from common.hailo_inference import HailoInfer
from common.toolbox import init_input_source, get_labels, load_json_file, preprocess, visualize, FrameRateTracker
from object_detection_post_process import inference_result_handler


import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the detection application.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Run object detection with optional tracking and performance measurement.")

    parser.add_argument(
        "-n", "--net",
        type=str,
        default="yolov8n.hef",
        help="Path to the network in HEF format."
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        default="bus.jpg",
        help="Path to the input (image, video, or folder)."
    )

    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=1,
        help="Number of images per batch."
    )

    parser.add_argument(
        "-l", "--labels",
        type=str,
        default=str(Path(__file__).parent.parent / "common" / "coco.txt"),
        help="Path to label file (e.g., coco.txt). If not set, default COCO labels will be used."
    )

    parser.add_argument(
        "-s", "--save_stream_output",
        action="store_true",
        help="Save the visualized stream output to disk."
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Directory to save result images or video."
    )

    parser.add_argument(
        "-r", "--resolution",
        type=str,
        choices=["sd", "hd", "fhd"],
        default="sd",
        help="(Camera only) Input resolution: 'sd' (640x480), 'hd' (1280x720), or 'fhd' (1920x1080)."
    )

    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable object tracking across frames."
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
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Labels file not found: {args.labels}")

    if args.output_dir is None:
        args.output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def run_inference_pipeline(net, input, batch_size, labels, output_dir,
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
    fps_tracker = None
    if show_fps:
        fps_tracker = FrameRateTracker()

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

    hailo_inference = HailoInfer(net, batch_size)
    height, width, _ = hailo_inference.get_input_shape()

    preprocess_thread = threading.Thread(
        target=preprocess, args=(images, cap, batch_size, input_queue, width, height)
    )
    postprocess_thread = threading.Thread(
        target=visualize, args=(output_queue, cap, save_stream_output,
                                output_dir, post_process_callback_fn, fps_tracker)
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
    run_inference_pipeline(args.net, args.input, args.batch_size, args.labels,
          args.output_dir, args.save_stream_output, args.resolution,
          args.track, args.show_fps)




if __name__ == "__main__":
    main()