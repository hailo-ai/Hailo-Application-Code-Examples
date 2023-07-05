#!/usr/bin/env python3

import numpy as np
from hailo_platform import (HEF, PcieDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                InputVStreamParams, OutputVStreamParams, FormatType)
from zenlog import log
from PIL import Image, ImageDraw, ImageFont
import os
import argparse

from hailo_model_zoo.core.postprocessing.detection.yolo import YoloPostProc
from hailo_model_zoo.core.postprocessing.detection.nanodet import NanoDetPostProc

# only for yolov8
class yolov8_class():
    def __init__(self):
        self.strides = [32,16,8]
        self.regression_length = 15
        self.scale_factors = [0, 0]
        self.device_pre_post_layers = device_pre_post_layers()
class device_pre_post_layers():
    def __init__(self):
        self.sigmoid = True
        

parser = argparse.ArgumentParser(description='Running a Hailo inference with actual images using Hailo API and OpenCV')
parser.add_argument('hef', help="HEF file path")
parser.add_argument('images', help="Images path to perform inference on. Could be either a single image or a folder containing the images")
parser.add_argument('arch', help="The architecture type of the model: yolo_v3, yolo_v4, yolov_4t (tiny-yolov4), yolo_v5, yolo_v5_nms, yolox, yolo_v6, yolo_v7 or yolo_v8.")
parser.add_argument('--class-num', help="The number of classes the model is trained on. Defaults to 80", default=80)
parser.add_argument('--labels', help="The path to the labels txt file. Should be in a form of NUM : LABEL. If no labels file is provided, no label will be added to the output")
args = parser.parse_args()


kwargs = {}
kwargs['device_pre_post_layers'] = None


## strides - Constant scalar per bounding box for scaling. One for each anchor by size of the anchor values
## sizes - The actual anchors for the bounding boxes
arch_dict = {'yolo_v3': 
                {'strides': [32,16,8], 
                 'sizes': np.array([[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]])},
             'yolo_v4': 
                 {'strides': [32,16,8], 
                  'sizes': np.array([[142, 110, 192, 243, 459, 401], [36, 75, 76, 55, 72, 146], [12, 16, 19, 36, 40, 28]])},
             'yolo_v4t': 
                 {'strides': [16,32], 
                  'sizes': np.array([[23, 27, 37, 58, 81, 82], [81, 82, 135, 169, 344, 319]])},
             'yolo_v5': 
                 {'strides': [32,16,8], 
                  'sizes': np.array([[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]])},
             'yolox': 
                 {'strides': [32,16,8], 
                  'sizes': np.array([[1, 1], [1, 1], [1, 1]])},
             'yolo_v7': 
                 {'strides': [32,16,8], 
                  'sizes': np.array([[142, 110, 192, 243, 459, 401], [36, 75, 76, 55, 72, 146], [12, 16, 19, 36, 40, 28]])},
             'yolo_v5_nms': {},
             'yolo_v8': {}}


# ---------------- Post-processing functions ----------------- #

def get_label(class_id):
    with open(args.labels, 'r') as f:
        labels = eval(f.read()) 
        return labels[class_id]

def draw_detection(draw, d, c, s, color, scale_factor):
    """Draw box and label for 1 detection."""
    if args.labels is not None:
        if args.arch == 'yolo_v8' or args.arch == 'yolo_v5_nms':
            label = get_label(c+1) + ": " + "{:.2f}".format(s) + '%'
        else:
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

# -------------- NMS on-chip\on-cpu functions --------------- #

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

# ---------------- Architecture functions ----------------- #

def postproc_yolov8(height, width, anchors, meta_arch, num_of_classes, raw_detections):
    raw_detections_keys = list(raw_detections.keys())
    raw_detections_keys.sort()
    
    yolov8_cls = yolov8_class()
    
    post_proc = NanoDetPostProc(img_dims=(height,width),
                                anchors=yolov8_cls, 
                                meta_arch=meta_arch, 
                                classes=num_of_classes,
                                nms_iou_thresh=0.7,
                                score_threshold=0.001,
                                **kwargs)
    
    layer_from_shape: dict = {raw_detections[key].shape:key for key in raw_detections_keys}

    detections = [raw_detections[layer_from_shape[1, 20, 20, 64]],
                    raw_detections[layer_from_shape[1, 20, 20, 80]],
                    raw_detections[layer_from_shape[1, 40, 40, 64]],
                    raw_detections[layer_from_shape[1, 40, 40, 80]],
                    raw_detections[layer_from_shape[1, 80, 80, 64]],
                    raw_detections[layer_from_shape[1, 80, 80, 80]]]    
   
    return post_proc.postprocessing(detections, device_pre_post_layers=yolov8_cls.device_pre_post_layers)

def postproc_yolov3(height, width, anchors, meta_arch, num_of_classes, raw_detections):
    raw_detections_keys = list(raw_detections.keys())
    raw_detections_keys.sort(reverse=True)
    
    post_proc = YoloPostProc(img_dims=(height,width), 
                            anchors=anchors,
                            meta_arch=meta_arch, 
                            classes=num_of_classes,
                            nms_iou_thresh=0.45,
                            score_threshold=0.01,
                            labels_offset=1,
                            **kwargs)
    
    detections = []
    for raw_det in raw_detections_keys:
        detections.append(raw_detections[raw_det])

    return post_proc.postprocessing(detections, **kwargs)
                    
def postproc_yolov4(height,width, anchors, meta_arch, num_of_classes, raw_detections):
    raw_detections_keys = list(raw_detections.keys())
    raw_detections_keys.sort()
    
    post_proc = YoloPostProc(img_dims=(height,width), 
                        anchors=anchors,
                        meta_arch=meta_arch, 
                        classes=num_of_classes,
                        nms_iou_thresh=0.45,
                        score_threshold=0.01,
                        labels_offset=1,
                        **kwargs)
    
    scales = [scale for scale in raw_detections_keys if 'scales' in scale]
    centers = [center for center in raw_detections_keys if 'centers' in center]
    probs = [prob for prob in raw_detections_keys if 'probs' in prob]
    objs = [obj for obj in raw_detections_keys if 'obj' in obj]

    # The detections that go in the postprocessing should have very specific order for the postprocessing to succeed.
    # It is reccomeneded NOT to change the below order
    detections = [raw_detections[centers[2]],
                    raw_detections[scales[2]],
                    raw_detections[objs[2]],
                    raw_detections[probs[2]],
                    raw_detections[centers[0]],
                    raw_detections[scales[0]],
                    raw_detections[objs[0]],
                    raw_detections[probs[0]],
                    raw_detections[centers[1]],
                    raw_detections[scales[1]],
                    raw_detections[objs[1]],
                    raw_detections[probs[1]]]
    
    return post_proc.postprocessing(detections, **kwargs)

def postproc_yolov4t(height, width, anchors, meta_arch, num_of_classes, raw_detections):
    raw_detections_keys = list(raw_detections.keys())
    raw_detections_keys.sort()
    
    post_proc = YoloPostProc(img_dims=(height,width), 
                        anchors=anchors,
                        meta_arch='yolo_v3', 
                        classes=num_of_classes,
                        nms_iou_thresh=0.3,
                        score_threshold=0.1,
                        labels_offset=0,
                        **kwargs)
    
    detections = []
    
    for raw_det in raw_detections_keys:
        detections.append(raw_detections[raw_det])
    
    return post_proc.postprocessing(detections, **kwargs)

def postproc_yolov5_yolov7(height,width, anchors, meta_arch, num_of_classes, raw_detections):
    raw_detections_keys = list(raw_detections.keys())
    raw_detections_keys.sort()
    
    post_proc = YoloPostProc(img_dims=(height,width), 
                            anchors=anchors,
                            meta_arch=meta_arch, 
                            classes=num_of_classes,
                            nms_iou_thresh=0.6,
                            score_threshold=0.001,
                            labels_offset=1, 
                            **kwargs)
    
    detections = []
    
    for raw_det in raw_detections_keys:
        detections.append(raw_detections[raw_det])
        
    return post_proc.postprocessing(detections, **kwargs)

  
def postproc_yolox_yolov6(height,width, anchors, meta_arch, num_of_classes, raw_detections):
    raw_detections_keys = list(raw_detections.keys())
    raw_detections_keys.sort()
    
    post_proc = YoloPostProc(img_dims=(height,width), 
                            anchors=anchors,
                            meta_arch=meta_arch, 
                            classes=num_of_classes,
                            nms_iou_thresh=0.65,
                            score_threshold=0.01,
                            labels_offset=1,
                            **kwargs)
    
    layer_from_shape: dict = {raw_detections[key].shape:key for key in raw_detections_keys}
    
    # The detections that go in the postprocessing should have very specific order. so,
    # we take the name of the layer name according to it's shape -  layer_from_shape(OUTPUT_SHAPE)-->LAYER_NAME
    detections = [raw_detections[layer_from_shape[1, 80, 80, 4]],
                    raw_detections[layer_from_shape[1, 80, 80, 1]],
                    raw_detections[layer_from_shape[1, 80, 80, 80]],
                    raw_detections[layer_from_shape[1, 40, 40, 4]],
                    raw_detections[layer_from_shape[1, 40, 40, 1]],
                    raw_detections[layer_from_shape[1, 40, 40, 80]],
                    raw_detections[layer_from_shape[1, 20, 20, 4]],
                    raw_detections[layer_from_shape[1, 20, 20, 1]],
                    raw_detections[layer_from_shape[1, 20, 20, 80]]]
    
    return post_proc.postprocessing(detections, **kwargs)

# ---------------- Pre-processing functions ----------------- #

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = image.size
    model_input_w, model_input_h = size
    scale = min(model_input_w / img_w, model_input_h / img_h)
    scaled_w = int(img_w * scale)
    scaled_h = int(img_h * scale)
    image = image.resize((scaled_w, scaled_h), Image.BICUBIC)
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

func_dict = {'yolo_v3': postproc_yolov3, 
             'yolo_v4': postproc_yolov4,
             'yolo_v4t': postproc_yolov4t,
             'yolo_v5': postproc_yolov5_yolov7,
             'yolox': postproc_yolox_yolov6,
             'yolo_v6': postproc_yolox_yolov6,
             'yolo_v7': postproc_yolov5_yolov7,
             'nanodet_v8': postproc_yolov8,
             }

images_path = args.images
num_of_classes = args.class_num

images = []

load_input_images(images_path, images)

anchors = {}
meta_arch = ''

arch = args.arch
arch_list = arch_dict.keys()

if arch in arch_list:
    anchors = arch_dict[arch]
    if arch == 'yolo_v7':
        meta_arch = 'yolo_v5'
    elif arch == 'yolo_v8':
        meta_arch = 'nanodet_v8'
    else:
        meta_arch = arch
else:
    if arch == 'yolo_v6':
        anchors = arch_dict['yolox']
        meta_arch = 'yolox'
    else:
        error = 'Not a valid architecture. Please choose an architecture from the this list: yolo_v3, yolo_v4, yolov_4t, yolo_v5, yolox, yolo_v6, yolo_v7, yolo_v8'
        raise ValueError(error)


devices = PcieDevice.scan_devices()
hef = HEF(args.hef)

inputs = hef.get_input_vstream_infos()
outputs = hef.get_output_vstream_infos()

with PcieDevice(devices[0]) as target:
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    [log.info('Input  layer: {} {}'.format(layer_info.name, layer_info.shape)) for layer_info in inputs]
    [log.info('Output layer: {} {}'.format(layer_info.name, layer_info.shape)) for layer_info in outputs]

    height, width, _ = hef.get_input_vstream_infos()[0].shape

    input_vstream_info = hef.get_input_vstream_infos()[0]

    input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)

    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        for i, image in enumerate(images):
            processed_image = preproc(image, height=height, width=width)
                        
            input_data = {input_vstream_info.name: np.expand_dims(processed_image, axis=0).astype(np.float32)}
            
            with network_group.activate(network_group_params):
                raw_detections = infer_pipeline.infer(input_data)
                
                if len(outputs) == 1 and 'nms' in outputs[0].name: 
                    results = post_nms_infer(raw_detections, outputs[0].name)
                else:
                    results = func_dict[meta_arch](height, width, anchors, meta_arch, int(num_of_classes), raw_detections)
                                        
                output_path = os.path.join(os.path.realpath('.'), 'output_images')
                if not os.path.isdir(output_path): 
                    os.mkdir(output_path)
                    
                img = letterbox_image(image, (width,height))
                                            
                post_process(results, img, i, output_path, width, height)
