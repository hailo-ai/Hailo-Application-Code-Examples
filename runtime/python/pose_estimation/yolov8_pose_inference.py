#!/usr/bin/env python3

import numpy as np
from zenlog import log
from PIL import Image
import os
import argparse
import cv2

from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                InputVStreamParams, OutputVStreamParams, FormatType)

from yolov8_pose_utils import *
        

parser = argparse.ArgumentParser(description='Running a Hailo inference with actual images using Hailo API and OpenCV')
parser.add_argument('hef', help="HEF file path")
parser.add_argument('images', help="Images path to perform inference on. Could be either a single image or a folder containing the images")
parser.add_argument('--class-num', help="The number of classes the model is trained on. Defaults to 1", default=1)
args = parser.parse_args()


kwargs = {
            'classes' : 1,
            'nms_max_output_per_class' : 300,
            'anchors' : {'regression_length' : 15, 'strides' : [8, 16, 32]},
            'score_threshold' : 0.001,
            'nms_iou_thresh' : 0.7,
            'meta_arch' : 'nanodet_v8',
            'device_pre_post_layers' : None
        }

# ---------------- Post-processing functions ----------------- #


def postproc_yolov8_pose(num_of_classes, raw_detections):

    raw_detections_keys = list(raw_detections.keys())
    layer_from_shape: dict = {raw_detections[key].shape:key for key in raw_detections_keys}
    
    detection_output_channels = (kwargs['anchors']['regression_length'] + 1) * 4 # (regression length + 1) * num_coordinates
    keypoints = 51

    endnodes = [raw_detections[layer_from_shape[1, 20, 20, detection_output_channels]],
                raw_detections[layer_from_shape[1, 20, 20, num_of_classes]],
                raw_detections[layer_from_shape[1, 20, 20, keypoints]],
                raw_detections[layer_from_shape[1, 40, 40, detection_output_channels]],
                raw_detections[layer_from_shape[1, 40, 40, num_of_classes]],
                raw_detections[layer_from_shape[1, 40, 40, keypoints]],
                raw_detections[layer_from_shape[1, 80, 80, detection_output_channels]],
                raw_detections[layer_from_shape[1, 80, 80, num_of_classes]],
                raw_detections[layer_from_shape[1, 80, 80, keypoints]]]
    
    predictions_dict = yolov8_pose_estimation_postprocess(endnodes, **kwargs)

    return predictions_dict

# ---------------- Pre-processing functions ----------------- #

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = image.size
    model_input_w, model_input_h = size
    scale = min(model_input_w / img_w, model_input_h / img_h)
    scaled_w = int(img_w * scale)
    scaled_h = int(img_h * scale)
    image = image.resize((scaled_w, scaled_h), Image.Resampling.BICUBIC)
    new_image = Image.new('RGB', size, (114,114,114))
    new_image.paste(image, ((model_input_w - scaled_w) // 2, (model_input_h - scaled_h) // 2))
    return new_image

def preproc(image, width=640, height=640, normalized=True):
    image = letterbox_image(image, (width, height))
    if normalized == False:
        ## normalized_image = (base - mean) / std, given mean=0.0, std=255.0
        image[:,:, 0] = image[:,:, 0] / 255.0
        image[:,:, 1] = image[:,:, 1] / 255.0
        image[:,:, 2] = image[:,:, 2] / 255.0
    
    return image

def load_input_images(images_path, images):
    # if running inference on a single image:
    if (images_path.endswith('.jpg') or images_path.endswith('.png') or images_path.endswith('.bmp') or images_path.endswith('.jpeg')):
        images.append(Image.open(images_path))
    # if running inference on an images directory:
    if (os.path.isdir(images_path)):
        for img in os.listdir(images_path):
            if (img.endswith(".jpg") or img.endswith(".png") or img.endswith('.bmp') or img.endswith('.jpeg')):
                images.append(Image.open(os.path.join(images_path, img)))
                
# ---------------- Start of the example --------------------- #

images_path = args.images
num_of_classes = args.class_num

images = []

load_input_images(images_path, images)

devices = Device.scan()
hef = HEF(args.hef)

inputs = hef.get_input_vstream_infos()
outputs = hef.get_output_vstream_infos()

with VDevice(device_ids=devices) as target:
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    [log.info('Input  layer: {} {}'.format(layer_info.name, layer_info.shape)) for layer_info in inputs]
    [log.info('Output layer: {} {}'.format(layer_info.name, layer_info.shape)) for layer_info in outputs]

    height, width, _ = hef.get_input_vstream_infos()[0].shape
    
    kwargs['img_dims'] = (height,width)

    input_vstream_info = hef.get_input_vstream_infos()[0]

    input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)

    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        for i, image in enumerate(images):
            processed_image = preproc(image, height=height, width=width)
                        
            input_data = {input_vstream_info.name: np.expand_dims(processed_image, axis=0).astype(np.float32)}
            
            with network_group.activate(network_group_params):
                raw_detections = infer_pipeline.infer(input_data)
                
                results = postproc_yolov8_pose(int(num_of_classes), raw_detections)
                                        
                output_path = os.path.join(os.path.realpath('.'), 'output_images')
                if not os.path.isdir(output_path): 
                    os.mkdir(output_path)
                
                image = Image.fromarray(cv2.cvtColor(visualize_pose_estimation_result(results, processed_image, **kwargs), cv2.COLOR_BGR2RGB))
                
                image.save(f'{output_path}/output_image{i}.jpg', 'JPEG')
                
                
