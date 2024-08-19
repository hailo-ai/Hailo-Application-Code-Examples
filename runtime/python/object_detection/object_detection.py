#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from loguru import logger
import queue
import threading
from object_detection_utils import ObjectDetectionUtils

# Add the parent directory to the system path to access utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference

IMAGE_EXTENSIONS = ('.jpg', '.png', '.bmp', '.jpeg')


def parse_args() -> argparse.Namespace:
    """
    Initialize argument parser for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Detection Example")
    parser.add_argument("-n", "--net", help="Path for the network in HEF format.", default="yolov7.hef")
    parser.add_argument("-i", "--input", default="zidane.jpg", help="Path to the input - either an image or a folder of images.")
    parser.add_argument("-b", "--batch_size", default=1, type=int, required=False, help="Number of images in one batch")
    parser.add_argument("-l", "--labels", default="coco.txt", help="Path to a text file containing labels. If no labels file is provided, coco2017 will be used.")
    
    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.net):
        raise FileNotFoundError(f"Network file not found: {args.net}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input path not found: {args.input}")
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Labels file not found: {args.labels}")

    return args


def load_input_images(images_path: str) -> list:
    """
    Load images from the specified path.

    Args:
        images_path (str): Path to the input image or directory of images.

    Returns:
        list: List of PIL.Image.Image objects.
    """
    path = Path(images_path)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [Image.open(path)]
    elif path.is_dir():
        return [Image.open(img) for img in path.glob("*") if img.suffix.lower() in IMAGE_EXTENSIONS]
    return []


def validate_images(images: list, batch_size: int) -> None:
    """
    Validate that images exist and are properly divisible by the batch size.

    Args:
        images (list): List of images.
        batch_size (int): Number of images per batch.

    Raises:
        ValueError: If images list is empty or not divisible by batch size.
    """
    if not images:
        raise ValueError('No valid images found in the specified path.')
    
    if len(images) % batch_size != 0:
        raise ValueError('The number of input images should be divisible by the batch size without any remainder.')


def divide_list_to_batches(images_list: list, batch_size: int):
    """
    Divide the list of images into batches.

    Args:
        images_list (list): List of images.
        batch_size (int): Number of images in each batch.

    Returns:
        generator: Generator yielding batches of images.
    """
    for i in range(0, len(images_list), batch_size):
        yield images_list[i: i + batch_size]


def enqueue_images(images: list, batch_size: int, input_queue: queue.Queue, width: int, height: int, utils: ObjectDetectionUtils) -> None:
    """
    Preprocess and enqueue images into the input queue as they are ready.

    Args:
        images (list): List of PIL.Image.Image objects.
        input_queue (queue.Queue): Queue for input images.
        width (int): Model input width.
        height (int): Model input height.
    """
    for batch in divide_list_to_batches(images, batch_size):
        processed_batch = []
        batch_array = []
        
        for image in batch:
            processed_image = utils.preprocess(image, width, height)
            processed_batch.append(processed_image)
            batch_array.append(np.array(processed_image))
        
        input_queue.put(processed_batch)  # Enqueue the batch of images and their arrays

    input_queue.put(None)  # Add sentinel value to signal end of input


def process_output(output_queue: queue.Queue, output_path: Path, width: int, height: int, utils: ObjectDetectionUtils) -> None:
    """
    Process and visualize the output results.

    Args:
        output_queue (queue.Queue): Queue for output results.
        output_path (Path): Path to save the output images.
        width (int): Image width.
        height (int): Image height.
    """
    image_id = 0
    while True:
        result = output_queue.get()
        if result is None:
            break  # Exit the loop if sentinel value is received
        
        processed_image, infer_results = result
        detections = utils.extract_detections(infer_results[0])
        utils.visualize(detections, processed_image, image_id, output_path, width, height)
        image_id += 1
    
    output_queue.task_done()  # Indicate that processing is complete


def infer(images: list, net_path: str, labels_path: str, batch_size: int, output_path: Path) -> None:
    """
    Initialize queues, HailoAsyncInference instance, and run the inference.

    Args:
        images (list): List of images to process.
        net_path (str): Path to the HEF model file.
        labels_path (str): Path to a text file containing labels.
        batch_size (int): Number of images per batch.
        output_path (Path): Path to save the output images.
    """
    utils = ObjectDetectionUtils(labels_path)
    
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    
    hailo_inference = HailoAsyncInference(net_path, input_queue, output_queue, batch_size)
    height, width, _ = hailo_inference.get_input_shape()

    enqueue_thread = threading.Thread(target=enqueue_images, args=(images, batch_size, input_queue, width, height, utils))
    process_thread = threading.Thread(target=process_output, args=(output_queue, output_path, width, height, utils))
    
    enqueue_thread.start()
    process_thread.start()
    
    hailo_inference.run()

    enqueue_thread.join()
    output_queue.put(None)  # Signal process thread to exit
    process_thread.join()

    hailo_inference.release_device()
    logger.info(f'Inference was successful! Results have been saved in {output_path}')


def main() -> None:
    """
    Main function to run the script.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Load input images
    images = load_input_images(args.input)
    
    # Validate images
    try:
        validate_images(images, args.batch_size)
    except ValueError as e:
        logger.error(e)
        return
    
    # Create output directory if it doesn't exist
    output_path = Path('output_images')
    output_path.mkdir(exist_ok=True)

    # Start the inference
    infer(images, args.net, args.labels, args.batch_size, output_path)


if __name__ == "__main__":
    main()
