import argparse
import os
import sys
from loguru import logger
import queue
import threading
from functools import partial
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.hailo_inference import HailoInfer
from common.toolbox import (
    init_input_source, 
    get_labels, 
    load_json_file, 
    preprocess, 
    visualize, 
    FrameRateTracker,
    resolve_net_arg,
    resolve_input_arg,
    resolve_output_resolution_arg,
    list_networks,
    list_inputs,
    oriented_object_detection_preprocess,
)
from oriented_object_detection_post_process import inference_result_handler

APP_NAME = Path(__file__).stem

def parse_args() -> argparse.Namespace:
    """
    Initialize argument parser for the script.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Instance segmentation supporting Yolov5, Yolov8, and FastSAM architectures.",
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
        "-l", "--labels",
        default=str(Path(__file__).parent.parent / "common" / "dota.txt"),
        help="Path to a text file containing labels. If not provided, dota_v1 will be used."
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Directory to save the results."
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
        help="Enable FPS performance measurement."
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


def run_inference_pipeline(
    net,
    input,
    batch_size,
    labels_file,
    output_dir,
    camera_resolution,
    output_resolution,
    framerate,
    save_stream_output=False,
    show_fps=False
) -> None:

    labels = get_labels(labels_file)
    # load local config.json from this example folder
    config_path = str(Path(__file__).parent / "config.json")
    config_data = load_json_file(config_path)

    cap, images = init_input_source(input, batch_size, camera_resolution)
    fps_tracker = None
    if show_fps:
        fps_tracker = FrameRateTracker()

    input_queue = queue.Queue()
    output_queue = queue.Queue()

    preprocess_callback_fn = partial(
        oriented_object_detection_preprocess,
        config_data=config_data,
    )
    
    post_process_callback_fn = partial(
        inference_result_handler, 
        labels=labels,
        config_data=config_data,
    )

    hailo_inference = HailoInfer(net, batch_size, input_type="UINT8", output_type="FLOAT32")
    height, width, _ = hailo_inference.get_input_shape()

    preprocess_thread = threading.Thread(
        target=preprocess, args=(images, cap, framerate, batch_size, input_queue, width, height, preprocess_callback_fn)
    )
    postprocess_thread = threading.Thread(
        target=visualize, args=(output_queue, cap, save_stream_output,
                                output_dir, post_process_callback_fn, fps_tracker, output_resolution, framerate)
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
    output_queue.put(None)
    postprocess_thread.join()

    if show_fps:
        logger.debug(fps_tracker.frame_rate_summary())

    logger.info('Oriented inference finished')


def infer(hailo_inference, input_queue, output_queue):
    while True:
        next_batch = input_queue.get()
        if not next_batch:
            break
        input_batch, preprocessed_batch = next_batch

        inference_callback_fn = partial(
            inference_callback,
            input_batch=input_batch,
            output_queue=output_queue
        )

        hailo_inference.run(preprocessed_batch, inference_callback_fn)

    hailo_inference.close()


def inference_callback(
    completion_info,
    bindings_list: list,
    input_batch: list,
    output_queue: queue.Queue
) -> None:
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
    args = parse_args()
    run_inference_pipeline(
        args.net,
        args.input,
        args.batch_size,
        args.labels,
        args.output_dir,
        args.camera_resolution,
        args.output_resolution,
        args.framerate,
        args.save_stream_output,
        args.show_fps
    )


if __name__ == "__main__":
    main()
