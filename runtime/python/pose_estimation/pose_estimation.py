#!/usr/bin/env python3

import os
import sys
import argparse
import multiprocessing as mp
from multiprocessing import Process
from pathlib import Path
from loguru import logger
from PIL import Image
from hailo_platform import HEF
from pose_estimation_utils import (output_data_type2dict,
                                   check_process_errors, PoseEstPostProcessing)

# Add the parent directory to the system path to access utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference, load_input_images, validate_images, divide_list_to_batches


def parse_args() -> argparse.Namespace:
    """
    Initialize argument parser for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Running a Hailo inference with actual images using Hailo API and OpenCV"
    )
    parser.add_argument(
        "-n", "--net",
        help="Path for the network in HEF format.",
        default="yolov8s_pose.hef"
    )
    parser.add_argument(
        "-i", "--input",
        default="zidane.jpg",
        help="Path to the input - either an image or a folder of images."
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

    args = parser.parse_args()
    # Validate paths
    if not os.path.exists(args.net):
        raise FileNotFoundError(f"Network file not found: {args.net}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input path not found: {args.input}")
    return args


def create_output_directory() -> Path:
    """
    Create the output directory if it does not exist.

    Returns:
        Path: Path object for the output directory.
    """
    output_path = Path('output_images')
    output_path.mkdir(exist_ok=True)
    return output_path


def preprocess_input(
    images: list[Image.Image],
    batch_size: int,
    input_queue: mp.Queue,
    width: int,
    height: int,
    post_processing: PoseEstPostProcessing
) -> None:
    """
    Preprocess and enqueue images into the input queue as they are ready.

    Args:
        images (list[Image.Image]): list of PIL.Image.Image objects.
        batch_size (int): Number of images in one batch.
        input_queue (mp.Queue): Queue for input images.
        width (int): Model input width.
        height (int): Model input height.
    """
    for batch in divide_list_to_batches(images, batch_size):
        processed_batch = []

        for image in batch:
            processed_image = post_processing.preprocess(image, width, height)
            processed_batch.append(processed_image)

        input_queue.put(processed_batch)

    input_queue.put(None)


def postprocess_output(
    output_queue: mp.Queue,
    output_path: Path,
    width: int,
    height: int,
    class_num: int,
   post_processing: PoseEstPostProcessing
) -> None:
    """
    Process and visualize the output results.

    Args:
        output_queue (mp.Queue): Queue for output results.
        output_path (Path): Path to save the output images.
        width (int): Image width.
        height (int): Image height.
        class_num (int): Number of classes.
        post_processing (PoseEstPostProcessing): Post-processing configuration.
    """
    image_id = 0
    while True:
        result = output_queue.get()
        if result is None:
            break  # Exit the loop if sentinel value is received

        processed_image, raw_detections = result
        post_processing.postprocess_and_visualize(processed_image, raw_detections,
                                    output_path, image_id, height, width, class_num)

        image_id += 1


def infer(
    images: list[Image.Image],
    net_path: str,
    batch_size: int,
    class_num: int,
    output_path: Path,
    data_type_dict: dict,
    post_processing: PoseEstPostProcessing
) -> None:
    """
    Run inference with HailoAsyncInference, handle processes, and ensure proper cleanup.

    Args:
        images (list[Image.Image]): List of images to process.
        net_path (str): Path to the HEF model file.
        batch_size (int): Number of images per batch.
        class_num (int): Number of classes.
        output_path (Path): Path to save the output images.
        data_type_dict (dict): Dictionary of layer names and data types.
        post_processing (PoseEstPostProcessing): Post-processing configuration.
    """
    input_queue = mp.Queue()
    output_queue = mp.Queue()

    hailo_inference = HailoAsyncInference(
        net_path, input_queue, output_queue, batch_size, output_type=data_type_dict
    )
    height, width, _ = hailo_inference.get_input_shape()

    preprocess = Process(
        target=preprocess_input,
        name="image_enqueuer",
        args=(images, batch_size, input_queue, width, height, post_processing)      
    )
    postprocess = Process(
        target=postprocess_output,
        name="image_processor",
        args=(
            output_queue, output_path, width, height, class_num, post_processing
        )
    )

    preprocess.start()
    postprocess.start()

    try:
        hailo_inference.run()
        preprocess.join()
        # To signal processing process to exit
        output_queue.put(None)
        postprocess.join()
      
        check_process_errors(preprocess, postprocess)
     
        logger.info(f'Inference was successful! Results have been saved in {output_path}')
      
    except Exception as e:
        logger.error(f"Inference error: {e}")
        # Ensure cleanup if there's an error
        input_queue.close()
        output_queue.close()
        preprocess.terminate()
        postprocess.terminate()

        os._exit(1)  # Force exit on error


def main() -> None:
    args = parse_args()
    images = load_input_images(args.input)

    try:
        validate_images(images, args.batch_size)
    except ValueError as e:
        logger.error(e)
        return

    output_path = create_output_directory()
    output_type_dict = output_data_type2dict(HEF(args.net), 'FLOAT32')

    post_processing = PoseEstPostProcessing(
        max_detections=300,
        score_threshold=0.001,
        nms_iou_thresh=0.7,
        regression_length=15,
        strides=[8, 16, 32]
    )

    infer(
        images, args.net, int(args.batch_size), int(args.class_num),
        output_path, output_type_dict, post_processing
    )

if __name__ == "__main__":
    main()
# End-of-file (EOF)
