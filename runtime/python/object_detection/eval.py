#!/usr/bin/env python3

import numpy as np
from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                InputVStreamParams, OutputVStreamParams, FormatType)
from zenlog import log
from PIL import Image, ImageDraw, ImageFont
import os
import argparse

def parse_args():
    """Initialize argument parser for the script."""
    parser = argparse.ArgumentParser(description="Detection Example - Tracker with ByteTrack and Supervision")
    parser.add_argument("-m", "--model", help="Path for the HEF model.", default="yolov5m_wo_spp_60p.hef")
    parser.add_argument("-i", "--input", default="input_example.mp4", help="Path to the input video.")
    parser.add_argument("-o", "--output", default="output_video.mp4", help="Path to the output video.")
    parser.add_argument("-b", "--batch", default=1, type=int, required=False, help="Number of images in one batch")
    parser.add_argument("-l", "--labels", default="coco.txt", help="Path to a text file containing labels. If no labels file is provided, coco2017 will be used.")
    parsed_args = parser.parse_args()
    return parsed_args

def get_label(class_id):
    with open(args.labels, 'r', encoding="utf-8") as f:
        class_names = f.read().splitlines()
    return class_names[class_id]

def draw_detection(draw, d, c, s, color, scale_factor):
    """Draw box and label for 1 detection."""
    if args.labels is not None:
        label = get_label(c) + ": " + "{:.2f}".format(s) + '%'
    else:
        label = ''
    ymin, xmin, ymax, xmax = d
    font = ImageFont.truetype("LiberationSans-Regular.ttf", size=15)
    draw.rectangle([(xmin * scale_factor, ymin * scale_factor), (xmax * scale_factor, ymax * scale_factor)], outline=color, width=2)
    draw.text((xmin * scale_factor + 4, ymin * scale_factor + 4), label, fill=color, font=font)
    return label

def post_process(detections, image, id, output_path, width, height, min_score=0.45, scale_factor=1):
    COLORS = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
    boxes = np.array(detections['detection_boxes'])[0]
    classes = np.array(detections['detection_classes'])[0].astype(int)
    scores = np.array(detections['detection_scores'])[0]
    draw = ImageDraw.Draw(image)
    if args.labels is not None:
        print(f'Labels detected in image{id}:')
    for idx in range(np.array(detections['num_detections'])[0]):
        if scores[idx] >= min_score:
            color = tuple(int(c) for c in COLORS[classes[idx]])
            scaled_box = [x*width if i%2 else x*height for i,x in enumerate(boxes[idx])]
            label = draw_detection(draw, scaled_box , classes[idx], scores[idx]*100.0, color, scale_factor)
            if args.labels is not None:
                print(label)
    image.save(f'{output_path}/output_image{id}.jpg', 'JPEG')

def extract_detections(input, boxes, scores, classes, num_detections, threshold=0.5):   
    for i, detection in enumerate(input):
        if len(detection) == 0:
            continue
        for j in range(len(detection)):
            bbox = np.array(detection)[j][:4]
            score = np.array(detection)[j][4]
            if score < threshold:
                continue
            else:
                boxes.append(bbox)
                scores.append(score)
                classes.append(i)
                num_detections = num_detections + 1
    return {'detection_boxes': [boxes], 
              'detection_classes': [classes], 
             'detection_scores': [scores],
             'num_detections': [num_detections]}

def post_nms_infer(raw_detections, input_name):
    boxes = []
    scores = []
    classes = []
    num_detections = 0
    
    detections = extract_detections(raw_detections[input_name][0], boxes, scores, classes, num_detections)
    
    return detections

def preprocess(image, model_w, model_h):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = image.size
    # Scale image
    scale = min(model_w / img_w, model_h / img_h)
    new_img_w = int(img_w * scale)
    new_img_h = int(img_h * scale)
    image = image.resize((new_img_w, new_img_h), Image.Resampling.BICUBIC)
    
    # Create a new padded image
    padded_image = Image.new('RGB', (model_w, model_h), (114,114,114))
    padded_image.paste(image, ((model_w - new_img_w) // 2, (model_h - new_img_h) // 2))
    return padded_image


def divide_list_to_batches(images_list, batch_size):
    for i in range(0, len(images_list), batch_size):
        yield images_list[i : i + batch_size]

def load_input_images(images_path, images):
    # if running inference on a single image:
    if (images_path.endswith('.jpg') or images_path.endswith('.png') or images_path.endswith('.bmp') or images_path.endswith('.jpeg')):
        images.append(Image.open(images_path))
    # if running inference on an images directory:
    if (os.path.isdir(images_path)):
        for img in os.listdir(images_path):
            if (img.endswith(".jpg") or img.endswith(".png") or img.endswith('.bmp') or img.endswith('.jpeg')):
                images.append(Image.open(os.path.join(images_path, img)))

if __name__ == "__main__":
    args = parse_args()
    images_path = args.input

    images = []

    load_input_images(images_path, images)

    devices = Device.scan()
    hef = HEF(args.model)

    inputs = hef.get_input_vstream_infos()
    outputs = hef.get_output_vstream_infos()
    batch_size = args.batch
    if len(images) % batch_size != 0:
        raise ValueError('The number of input images should be divisiable by the batch size without any remainder. Please either change the batch size to divide the number of images with no remainder or change the number of images')

    with VDevice(device_ids=devices) as target:
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_name = hef.get_network_group_names()[0]
        configure_params[network_name].batch_size = batch_size
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        [log.info('Input  layer: {} {}'.format(layer_info.name, layer_info.shape)) for layer_info in inputs]
        [log.info('Output layer: {} {}'.format(layer_info.name, layer_info.shape)) for layer_info in outputs]

        height, width, _ = hef.get_input_vstream_infos()[0].shape

        input_vstream_info = hef.get_input_vstream_infos()[0]

        input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)

        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            batched_images = list(divide_list_to_batches(images, batch_size))
            for batch_idx, batch_images in enumerate(batched_images):
                processed_input_images = []
                
                for i, image in enumerate(batch_images):
                    processed_image = preprocess(image, width, height)
                    processed_input_images.append(np.array(processed_image))

                with network_group.activate(network_group_params):
                    raw_detections = infer_pipeline.infer(np.array(processed_input_images).astype(np.float32))

                    results = post_nms_infer(raw_detections, outputs[0].name)

                    output_path = os.path.join(os.path.realpath('.'), 'output_images')
                    if not os.path.isdir(output_path):
                        os.mkdir(output_path)

                    for j in range(len(batch_images)):
                        img = preprocess(batch_images[j], width, height)
                        post_process(results, img, (batch_idx*len(batch_images))+j, output_path, width, height)

