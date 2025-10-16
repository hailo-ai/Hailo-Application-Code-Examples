#!/usr/bin/bash

if [[ "$1" == "--arch" && "$2" == "8" ]]; then
    wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/images/dog_bicycle.jpg
    wget https://ultralytics.com/images/zidane.jpg
    wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov5m_seg.hef
    wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8/yolov8s_seg.hef
    wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8/fast_sam_s.hef
    wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8/yolov5m_seg_with_nms.hef
elif [[ "$1" == "--arch" && "$2" == "10" ]]; then
    wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/images/dog_bicycle.jpg
    wget https://ultralytics.com/images/zidane.jpg
    wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/yolov5n_seg.hef
    wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.0.0/hailo15h/yolov8s_seg.hef
else
    echo "Usage: $0 --arch 8    # for Hailo-8"
    echo "       $0 --arch 10   # for Hailo-10"
    exit 1
fi