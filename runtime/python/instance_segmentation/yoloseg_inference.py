#!/usr/bin/env python3

import numpy as np
from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                InputVStreamParams, OutputVStreamParams, FormatType)
from zenlog import log
from PIL import Image
import os
import argparse

from hailo_model_zoo.core.postprocessing.instance_segmentation_postprocessing import yolov5_seg_postprocess
from hailo_model_zoo.core.postprocessing.instance_segmentation_postprocessing import yolov8_seg_postprocess
from hailo_model_zoo.core.postprocessing.instance_segmentation_postprocessing import visualize_yolov5_seg_results # Uses also for the v8 and fast_sam visualization
        

parser = argparse.ArgumentParser(description='Running a Hailo inference with actual images using Hailo API and OpenCV')
parser.add_argument('hef', help="HEF file path")
parser.add_argument('images', help="Images path to perform inference on. Could be either a single image or a folder containing the images")
parser.add_argument('arch', help="The architecture type of the model: v5, v8 or fast")
parser.add_argument('--class-num', help="The number of classes the model is trained on. Defaults to 80 for v5 and v8, and 1 for fast_sam.", default=80)
parser.add_argument('--output_dir', help="The path to the output directory where the images would be save. Default to the output_images folder in currect directory.")
args = parser.parse_args()


kwargs = {}
kwargs['device_pre_post_layers'] = None


## strides - Constant scalar per bounding box for scaling. One for each anchor by size of the anchor values
## sizes - The actual anchors for the bounding boxes
## regression_length - The regression length required for the distance estimation
arch_dict = {'v5':
                { 'anchors': 
                 {'strides': [32,16,8], 
                  'sizes': np.array([[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]])}
                },
             'v8': {'anchors':
                 {'strides': [32,16,8], 
                  'regression_length' : 15}
                 },
            'fast': {'anchors':
                 {'strides': [32,16,8], 
                  'regression_length' : 15}}
            }


# --------------------------------------------------------- #

# ---------------- Architecture functions ----------------- #

def postproc_yolov8seg(raw_detections):
    
    raw_detections_keys = list(raw_detections.keys())
    layer_from_shape: dict = {raw_detections[key].shape:key for key in raw_detections_keys}
    
    mask_channels = 32
    
    detection_output_channels = (kwargs['anchors']['regression_length'] + 1) * 4 # (regression length + 1) * num_coordinates
    
    endnodes = [raw_detections[layer_from_shape[1, 80, 80, detection_output_channels]],
                raw_detections[layer_from_shape[1, 80, 80, kwargs['classes']]],
                raw_detections[layer_from_shape[1, 80, 80, mask_channels]],
                raw_detections[layer_from_shape[1, 40, 40, detection_output_channels]],
                raw_detections[layer_from_shape[1, 40, 40, kwargs['classes']]],
                raw_detections[layer_from_shape[1, 40, 40, mask_channels]],
                raw_detections[layer_from_shape[1, 20, 20, detection_output_channels]],
                raw_detections[layer_from_shape[1, 20, 20, kwargs['classes']]],
                raw_detections[layer_from_shape[1, 20, 20, mask_channels]],
                raw_detections[layer_from_shape[1, 160, 160, mask_channels]]]
    
    
    predictions_dict = yolov8_seg_postprocess(endnodes, **kwargs)
    
    return predictions_dict
                    

def postproc_yolov5seg(raw_detections):
        
    raw_detections_keys = list(raw_detections.keys())
    layer_from_shape: dict = {raw_detections[key].shape:key for key in raw_detections_keys}
    
    mask_channels = 32
    
    detection_channels = (kwargs['classes'] + 4 + 1 + mask_channels) *  len(kwargs['anchors']['strides']) # (num_classes + num_coordinates + objectness + mask) * strides_list_len 
    
    endnodes = [raw_detections[layer_from_shape[1, 160, 160, mask_channels]],
                raw_detections[layer_from_shape[1, 80, 80, detection_channels]],
                raw_detections[layer_from_shape[1, 40, 40, detection_channels]],
                raw_detections[layer_from_shape[1, 20, 20, detection_channels]]]
    
    predictions_dict = yolov5_seg_postprocess(endnodes, **kwargs)
    
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
        image = np.array(image)
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

func_dict = {
            'yolov5_seg': postproc_yolov5seg,
            'yolov8_seg': postproc_yolov8seg,
            'fast_sam': postproc_yolov8seg,
            }

images_path = args.images

images = []

load_input_images(images_path, images)

anchors = {}
meta_arch = ''

arch = args.arch
arch_list = arch_dict.keys()

num_of_classes = args.class_num

if arch in arch_list:
    anchors = arch_dict[arch]
    kwargs['anchors'] = arch_dict[arch]['anchors']
    if arch == 'v5':
        meta_arch = 'yolov5_seg'
        kwargs['score_threshold'] = 0.001
        kwargs['nms_iou_thresh'] = 0.6
    if arch == 'v8':
        meta_arch = 'yolov8_seg'
        kwargs['score_threshold'] = 0.001
        kwargs['nms_iou_thresh'] = 0.7
        kwargs['meta_arch'] = 'yolov8_seg_postprocess'
    if arch == 'fast':
        meta_arch = 'fast_sam'
        kwargs['score_threshold'] = 0.25
        kwargs['nms_iou_thresh'] = 0.7
        kwargs['meta_arch'] = 'yolov8_seg_postprocess'
        num_of_classes = '1'
        kwargs['classes'] = 1
        
else:
    error = 'Not a valid architecture. Please choose an architecture from the this list: v5, v8, fast'
    raise ValueError(error)


kwargs['classes'] = int(num_of_classes)

output_dir = args.output_dir

if not output_dir:
    output_dir = 'output_images'

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
                
                results = func_dict[meta_arch](raw_detections)[0]
                                        
                output_path = os.path.join(os.path.realpath('.'), 'output_images')
                if not os.path.isdir(output_path): 
                    os.mkdir(output_path)
                                                                       
                processed_img = Image.fromarray(visualize_yolov5_seg_results(results, np.expand_dims(np.array(processed_image), axis=0), score_thres=0.3, class_names=num_of_classes, **kwargs))
                
                processed_img.save(f'{output_dir}/output_image{i}.jpg', 'JPEG')
