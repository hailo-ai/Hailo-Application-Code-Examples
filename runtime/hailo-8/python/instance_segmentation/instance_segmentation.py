#!/usr/bin/env python3
import argparse
import os
import sys
from loguru import logger
import queue
import threading
from types import SimpleNamespace
from post_process.postprocessing import inference_result_handler
from functools import partial
import time
from pathlib import Path
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.tracker.byte_tracker import BYTETracker
from common.hailo_inference import HailoAsyncInference
from common.toolbox import init_input_source, load_json_file, get_labels, visualize, preprocess
frame_counter = [0]


def parse_args() -> argparse.Namespace:
    """
    Initialize argument parser for the script.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Instance segmentation supporting Yolov5, Yolov8, and FastSAM architectures."
    )

    parser.add_argument(
        "-n", "--net",
        help="Path for the network in HEF format.",
        required=True
    )
    parser.add_argument(
        "-i", "--input",
        default="zidane.jpg",
        help="Path to the input - either an image or a folder of images."
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=1,
        help="Number of images in one batch"
    )
    parser.add_argument(
        "-s", "--save_stream_output",
        action="store_true",
        help="Save the output of the inference from a stream."
    )
    parser.add_argument(
        "-a", "--arch",
        required=True,
        help="The architecture type of the model: v5, v8 or fast"
    )
    parser.add_argument(
        "-l", "--labels",
        default=str(Path(__file__).parent.parent / "common" / "coco.txt"),
        help="Path to a text file containing labels. If not provided, coco2017 will be used."
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Directory to save the results."
    )
    parser.add_argument(
        "-r", "--resolution",
        choices=["sd", "hd", "fhd"],
        default=None,
        help="(Camera input only) Choose output resolution: 'sd' (640x480), 'hd' (1280x720), or 'fhd' (1920x1080). "
             "If not specified, the camera's native resolution will be used."
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable object tracking across frames."
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Enable FPS performance measurement."
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


def infer(
    net,
    input_path,
    arch,
    batch_size,
    labels_file,
    output_dir,
    save_stream_output=False,
    resolution="sd",
    enable_tracking=False,
    show_fps=False
) -> None:
    """
    Initialize queues, HailoAsyncInference instance, and run the inference.
    """
    config_data = load_json_file("config.json")
    labels = get_labels(labels_file)

    # Initialize input source from string: "camera", video file, or image folder
    cap, images = init_input_source(input_path, batch_size, resolution)
    tracker = None

    if enable_tracking:
        # Load tracker config from config_data
        tracker_config = config_data.get("visualization_params", {}).get("tracker", {})
        tracker = BYTETracker(SimpleNamespace(**tracker_config))

    input_queue = queue.Queue()
    output_queue = queue.Queue()

    inference_callback_fn = partial(inference_callback, output_queue=output_queue)

    hailo_inference = HailoAsyncInference(
        net,
        input_queue,
        inference_callback_fn,
        batch_size,
        output_type="FLOAT32",
        send_original_frame=True
    )

    post_process_callback_fn = partial(
        inference_result_handler,
        tracker=tracker,
        config_data=config_data,
        arch=arch,
        labels=labels,
        nms_postprocess_enabled=hailo_inference.is_nms_postprocess_enabled()
    )

    height, width, _ = hailo_inference.get_input_shape()

    preprocess_thread = threading.Thread(
        target=preprocess,
        args=(images, cap, batch_size, input_queue, width, height)
    )

    postprocess_thread = threading.Thread(
        target=visualize,
        args=(output_queue, cap, save_stream_output, output_dir, post_process_callback_fn, frame_counter)
    )

    if show_fps:
        start_time = time.time()

    preprocess_thread.start()
    postprocess_thread.start()
    hailo_inference.run()

    preprocess_thread.join()
    output_queue.put(None)  # Signal process thread to exit
    postprocess_thread.join()

    logger.info("Inference was successful!")

    if show_fps:
        end_time = time.time()
        fps = frame_counter[0] / (end_time - start_time)
        logger.debug(f"Processed {frame_counter[0]} frames at {fps:.2f} FPS")


def main() -> None:
    args = parse_args()
    infer(
        args.net,
        args.input,
        args.arch,
        args.batch_size,
        args.labels,
        args.output_dir,
        args.save_stream_output,
        args.resolution,
        args.track,
        args.show_fps
    )


if __name__ == "__main__":
    main()
