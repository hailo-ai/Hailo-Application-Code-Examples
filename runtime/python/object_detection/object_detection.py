#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from loguru import logger
import queue
import threading

# Add the parent directory to the system path to access utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference

IMAGE_EXTENSIONS = ('.jpg', '.png', '.bmp', '.jpeg')
LABEL_FONT = "LiberationSans-Regular.ttf"
PADDING_COLOR = (114, 114, 114)
COLORS = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

def parse_args():
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
    
    return parser.parse_args()


def get_labels(labels_path):
    """
    Load labels from a file.

    Args:
        labels_path (str): Path to the labels file.

    Returns:
        list: List of class names.
    """
    with open(labels_path, 'r', encoding="utf-8") as f:
        class_names = f.read().splitlines()
    return class_names


def draw_detection(labels, draw, box, cls, score, color, scale_factor):
    """
    Draw box and label for one detection.

    Args:
        labels (list): List of class labels.
        draw (ImageDraw.Draw): Draw object to draw on the image.
        box (list): Bounding box coordinates.
        cls (int): Class index.
        score (float): Detection score.
        color (tuple): Color for the bounding box.
        scale_factor (float): Scale factor for coordinates.

    Returns:
        str: Detection label.
    """
    label = f"{labels[cls]}: {score:.2f}%"
    ymin, xmin, ymax, xmax = box
    font = ImageFont.truetype(LABEL_FONT, size=15)
    draw.rectangle([(xmin * scale_factor, ymin * scale_factor), (xmax * scale_factor, ymax * scale_factor)], outline=color, width=2)
    draw.text((xmin * scale_factor + 4, ymin * scale_factor + 4), label, fill=color, font=font)


def visualize(labels, detections, image, image_id, output_path, width, height, min_score=0.45, scale_factor=1):
    """
    Visualize detections on the image.

    Args:
        labels (list): List of class labels.
        detections (dict): Detection results.
        image (PIL.Image.Image): Image to draw on.
        image_id (int): Image identifier.
        output_path (Path): Path to save the output image.
        width (int): Image width.
        height (int): Image height.
        min_score (float): Minimum score threshold.
        scale_factor (float): Scale factor for coordinates.
    """
    boxes = detections['detection_boxes']
    classes = detections['detection_classes']
    scores = detections['detection_scores']
    draw = ImageDraw.Draw(image)

    for idx in range(detections['num_detections']):
        if scores[idx] >= min_score:
            color = tuple(int(c) for c in COLORS[classes[idx]])
            scaled_box = [x * width if i % 2 == 0 else x * height for i, x in enumerate(boxes[idx])]
            draw_detection(labels, draw, scaled_box, classes[idx], scores[idx] * 100.0, color, scale_factor)
            
    image.save(output_path / f'output_image{image_id}.jpg', 'JPEG')


def extract_detections(input_data, threshold=0.5):
    """
    Extract detections from the input data.

    Args:
        input_data (list): Raw detections from the model.
        threshold (float): Score threshold for filtering detections.

    Returns:
        dict: Filtered detection results.
    """
    boxes, scores, classes = [], [], []
    num_detections = 0
    
    for i, detection in enumerate(input_data):
        if len(detection) == 0:
            continue

        for det in detection:
            bbox, score = det[:4], det[4]

            if score >= threshold:
                boxes.append(bbox)
                scores.append(score)
                classes.append(i)
                num_detections += 1
                
    return {
        'detection_boxes': boxes, 
        'detection_classes': classes, 
        'detection_scores': scores,
        'num_detections': num_detections
        }


def preprocess(image, model_w, model_h):
    """
    Resize image with unchanged aspect ratio using padding.

    Args:
        image (PIL.Image.Image): Input image.
        model_w (int): Model input width.
        model_h (int): Model input height.

    Returns:
        PIL.Image.Image: Preprocessed and padded image.
    """
    img_w, img_h = image.size
    # Scale image
    scale = min(model_w / img_w, model_h / img_h)
    new_img_w, new_img_h = int(img_w * scale), int(img_h * scale)
    image = image.resize((new_img_w, new_img_h), Image.Resampling.BICUBIC)
    
    # Create a new padded image
    padded_image = Image.new('RGB', (model_w, model_h), PADDING_COLOR)
    padded_image.paste(image, ((model_w - new_img_w) // 2, (model_h - new_img_h) // 2))
    return padded_image


def load_input_images(images_path):
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


def divide_list_to_batches(images_list, batch_size):
    """
    Divide the list of images into batches.

    Args:
        images_list (list): List of images.
        batch_size (int): Number of images in each batch.

    Returns:
        generator: Generator yielding batches of images.
    """
    for i in range(0, len(images_list), batch_size):
        yield images_list[i : i + batch_size]


def enqueue_images(images, batch_size, input_queue, width, height):
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
            processed_image = preprocess(image, width, height)
            processed_batch.append(processed_image)
            batch_array.append(np.array(processed_image))
        
        input_queue.put(processed_batch)  # Enqueue the batch of images and their arrays

    input_queue.put(None)  # Add sentinel value to signal end of input

def process_output(output_queue, labels, output_path, width, height):
    """
    Process and visualize the output results.

    Args:
        output_queue (queue.Queue): Queue for output results.
        labels (list): List of class labels.
        output_path (Path): Path to save the output images.
        width (int): Image width.
        height (int): Image height.
    """
    image_id = 0
    while True:
        result = output_queue.get()
        if result is None:
            break
        
        processed_image, infer_results = result
        detections = extract_detections(infer_results)
        visualize(labels, detections, processed_image, image_id, output_path, width, height)
        image_id += 1

def infer(images, net_path, labels, batch_size, output_path):
    """
    Initialize queues, HailoAsyncInference instance, and run the inference.

    Args:
        images (list): List of images to process.
        net_path (str): Path to the HEF model file.
        labels (list): List of class labels.
        batch_size (int): Number of images per batch.
        output_path (Path): Path to save the output images.
    """
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    
    hailo_inference = HailoAsyncInference(net_path, input_queue, output_queue, batch_size)
    height, width, _ = hailo_inference.get_input_shape()

    # Start the enqueueing and processing threads
    enqueue_thread = threading.Thread(target=enqueue_images, args=(images, batch_size, input_queue, width, height))
    process_thread = threading.Thread(target=process_output, args=(output_queue, labels, output_path, width, height))
    
    enqueue_thread.start()
    process_thread.start()
    
    # Start asynchronous inference
    hailo_inference.run()

    # Wait for threads to finish
    enqueue_thread.join()
    output_queue.put(None)  # Signal to the processing thread to stop
    process_thread.join()

    hailo_inference.release_device()
    logger.info(f'Inference was successful! Results have been saved in {output_path}')

def main():
    """
    Main function to run the script.
    """
    # Parse command line arguments
    args = parse_args()
    images = load_input_images(args.input)
    
    if not images:
        logger.error('No valid images found in the specified path.')
        return

    if len(images) % args.batch_size != 0:
        logger.error('The number of input images should be divisible by the batch size without any remainder. '
                     'Please either change the batch size to divide the number of images with no remainder or '
                     'change the number of images.')
        return
    
    # Create output directory if it doesn't exist
    output_path = Path('output_images')
    output_path.mkdir(exist_ok=True)
    
    labels = get_labels(args.labels)

    # Run the inference process
    infer(images, args.net, labels, args.batch_size, output_path)

if __name__ == "__main__":
    main()

