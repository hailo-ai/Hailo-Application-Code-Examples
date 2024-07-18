#!/usr/bin/env python3

import cv2
import os, random, time
import numpy as np
from multiprocessing import Process
import yolox_stream_report_detections as report
from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)

# yolox_s_leaky input resolution
INPUT_RES_H = 640
INPUT_RES_W = 640

# Loading compiled HEFs to device:
model_name = 'yolox_s_leaky'
hef_path = 'resources/hefs/{}.hef'.format(model_name)
video_dir = './resources/video/'
hef = HEF(hef_path)
mp4_files = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f)) and f.endswith('.mp4')]

devices = Device.scan()

with VDevice(device_ids=devices) as target:
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()
        input_vstream_info = hef.get_input_vstream_infos()[0]
        output_vstream_info = hef.get_output_vstream_infos()[0]
        input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
        height, width, channels = hef.get_input_vstream_infos()[0].shape
                            
        source = 'camera'
        cap = cv2.VideoCapture(0)

        # check if the camera was opened successfully
        if not cap.isOpened():
            print("Could not open camera")
            exit()

        while True:
            # read a frame from the video source
            ret, frame = cap.read()

            # Get height and width from capture
            orig_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  
            orig_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)        

            # check if the frame was successfully read
            if not ret:
                print("Could not read frame")
                break

            # loop if video source
            if source == 'video' and not cap.get(cv2.CAP_PROP_POS_FRAMES) % cap.get(cv2.CAP_PROP_FRAME_COUNT):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # resize image for yolox_s_leaky input resolution and infer it
            resized_img = cv2.resize(frame, (INPUT_RES_H, INPUT_RES_W), interpolation = cv2.INTER_AREA)
            with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                input_data = {input_vstream_info.name: np.expand_dims(np.asarray(resized_img), axis=0).astype(np.float32)}    
                with network_group.activate(network_group_params):
                    infer_results = infer_pipeline.infer(input_data)
            
            # create dictionary that returns layer name from tensor shape (required for postprocessing)
            layer_from_shape: dict = {infer_results[key].shape:key for key in infer_results.keys()}            

            from hailo_model_zoo.core.postprocessing.detection import yolo
            # postprocessing info for constructor as recommended in hailo_model_zoo/cfg/base/yolox.yaml
            anchors = {"strides": [32, 16, 8], "sizes": [[1, 1], [1, 1], [1, 1]]}
            yolox_post_proc = yolo.YoloPostProc(img_dims=(INPUT_RES_H,INPUT_RES_W), nms_iou_thresh=0.65, score_threshold=0.01, 
                                                anchors=anchors, output_scheme=None, classes=80, labels_offset=1, 
                                                meta_arch="yolox", device_pre_post_layers=[])                

            # Order of insertion matters since we need the reorganized tensor to be in (BS,H,W,85) shape
            endnodes = [infer_results[layer_from_shape[1, 80, 80, 4]],  # stride 8 
                        infer_results[layer_from_shape[1, 80, 80, 1]],  # stride 8 
                        infer_results[layer_from_shape[1, 80, 80, 80]], # stride 8 
                        infer_results[layer_from_shape[1, 40, 40, 4]],  # stride 16
                        infer_results[layer_from_shape[1, 40, 40, 1]],  # stride 16
                        infer_results[layer_from_shape[1, 40, 40, 80]], # stride 16
                        infer_results[layer_from_shape[1, 20, 20, 4]],  # stride 32
                        infer_results[layer_from_shape[1, 20, 20, 1]],  # stride 32
                        infer_results[layer_from_shape[1, 20, 20, 80]]  # stride 32
                    ]
            hailo_preds = yolox_post_proc.yolo_postprocessing(endnodes)
            num_detections = int(hailo_preds['num_detections'])
            scores = hailo_preds["detection_scores"][0].numpy()
            classes = hailo_preds["detection_classes"][0].numpy()
            boxes = hailo_preds["detection_boxes"][0].numpy()
            if scores[0] == 0:
                num_detections = 0
            preds_dict = {'scores': scores, 'classes': classes, 'boxes': boxes, 'num_detections': num_detections}
            frame = report.report_detections(preds_dict, frame, scale_factor_x = orig_w, scale_factor_y = orig_h)
            cv2.imshow('frame', frame)
            
            # wait for a key event
            key = cv2.waitKey(1)

            # switch between camera and video source
            if key == ord('c'):
                source = 'camera'
                cap.release()
                cap = cv2.VideoCapture(0)
            elif key == ord('v'):
                source = 'video'
                cap.release()
                random_mp4 = random.choice(mp4_files)
                cap = cv2.VideoCapture(video_dir+random_mp4)
            elif key == ord('q'):
                break

# release the video source and destroy all windows
cap.release()
cv2.destroyAllWindows()