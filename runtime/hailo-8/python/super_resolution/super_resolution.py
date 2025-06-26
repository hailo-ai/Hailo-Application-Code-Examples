#!/usr/bin/env python3
import numpy as np
from PIL import Image
from pathlib import Path
import os
from loguru import logger
import argparse
import sys
from typing import List
import threading
import queue
from super_resolution_utils import SrganUtils, Espcnx4Utils, SuperResolutionUtils
from functools import partial

# Add the parent directory to the system path to access utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.hailo_inference import HailoAsyncInference
from common.toolbox import load_input_images, validate_images, divide_list_to_batches

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
        "-i", "--input", 
        default="input_image.png",
        help="Path to the input - either an image or a folder of images."
    )
    parser.add_argument(
        "-b", "--batch_size", 
        default=1,
        type=int,
        required=False,
        help="Number of images in one batch"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output_images",
        help="Path of folder for output images",
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.net):
        raise FileNotFoundError(f"Network file not found: {args.net}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input path not found: {args.input}")

    return args

def enqueue_images(
    images: List[Image.Image],
    batch_size: int,
    input_queue: queue.Queue,
    width: int,
    height: int,
    utils: SuperResolutionUtils,
) -> None:
    """
    Preprocess and enqueue images into the input queue as they are ready.

    Args:
        images (List[Image.Image]): List of PIL.Image.Image objects.
        batch_size (int): Number of images per batch.
        input_queue (queue.Queue): Queue for input images.
        width (int): Model input width.
        height (int): Model input height.
        utils (SuperResolutionUtils): Utility class for super resolution preprocessing.
    """
    for batch in divide_list_to_batches(images, batch_size):
        processed_batch = []

        for image in batch:
            processed_image = utils.pre_process(image, width, height)
            processed_batch.append(processed_image)

        input_queue.put(processed_batch)

    input_queue.put(None)

def process_output(
    output_queue: queue.Queue,
    input_images: List[Image.Image],
    utils: SuperResolutionUtils,
    results: List[Image.Image], 
) -> None:
    """
    Process and visualize the output results.

    Args:
        output_queue (queue.Queue): Queue for output results.
        input_images (List[Image.Image]): List of input images.
        utils (SuperResolutionUtils): Utility class for super resolution visualization.
        results (List[Image.Image]): List to store the processed output images.
    """
    image_id = 0
    while True:
        result = output_queue.get()
        if result is None:
            break  # Exit the loop if sentinel value is received

        _, infer_results = result

        # Deals with the expanded results from hailort versions < 4.19.0
        if len(infer_results) == 1:
            infer_results = infer_results[0]

        infer_results = utils.post_process(infer_results, input_images[image_id])
        image_id += 1
        results.append(infer_results)

    output_queue.task_done()  # Indicate that processing is complete



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
    input_images: List[Image.Image],
    net_path: str,
    batch_size: int,
    output_path: Path,
) -> None:
    """
    Initialize queues, HailoAsyncInference instance, and run the inference.

    Args:
        input_images (List[Image.Image]): List of images to process.
        net_path (str): Path to the HEF model file.
        batch_size (int): Number of images per batch.
        output_path (Path): Path to save the output images.
    """
    utils = None
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    results = [] 

    inference_callback_fn = partial(inference_callback, output_queue=output_queue)

    if 'espcn' in net_path:
        utils = Espcnx4Utils()
        hailo_inference = HailoAsyncInference(net_path, input_queue, inference_callback_fn, batch_size, input_type="FLOAT32", output_type="FLOAT32")
    else:
        utils = SrganUtils()
        hailo_inference = HailoAsyncInference(net_path, input_queue, inference_callback_fn, batch_size)
    
    height, width, _ = hailo_inference.get_input_shape()
    enqueue_thread = threading.Thread(
        target=enqueue_images, 
        args=(input_images, batch_size, input_queue, width, height, utils)
    )
    process_thread = threading.Thread(
        target=process_output, 
        args=(output_queue, input_images, utils, results)
    )

    enqueue_thread.start()
    process_thread.start()

    hailo_inference.run()

    enqueue_thread.join()
    output_queue.put(None)  # Signal process thread to exit
    process_thread.join()

    # Save the results
    save_results(input_images, results, output_path)

    logger.info(
        f'Inference was successful! Results have been saved in {output_path}'
    )

def save_results(images: List[Image.Image], results: List[Image.Image], output_path: Path) -> None:    
    """
    Save the results of the inference to the output path.

    Args:
        images (List[Image.Image]): List of PIL.Image.Image objects.
        results (List[Image.Image]): List of PIL.Image.Image objects.
        output_path (Path): Path to save the output images.
    """
    if results:
        width, height = results[0].size
    for idx, (image, result) in enumerate(zip(images, results)):
        image = image.resize((width, height), Image.Resampling.BICUBIC)
        result.save(output_path / f"sr_output_{idx}.png")
        Image.fromarray(np.hstack((np.array(image), np.array(result)))).save(output_path / f"comparison_{idx}.png")

def main() -> None:
    """
    Main function to run the script.
    """
    # Parse command line arguments
    args = parse_args()

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)

    # Load input images
    images = load_input_images(args.input)

    # Validate images
    try:
        validate_images(images, args.batch_size)
    except ValueError as e:
        logger.error(e)
        return

    # Start the inference
    infer(images, args.net, args.batch_size, output_path)

if __name__ == "__main__":
    main()
