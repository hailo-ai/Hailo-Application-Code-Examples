#!/usr/bin/env bash

if [[ "$1" == "--arch" && "$2" == "8" ]]; then
    wget https://ultralytics.com/images/zidane.jpg
    wget -https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.12.0/hailo8/yolov8s_pose.hef
elif [[ "$1" == "--arch" && "$2" == "10" ]]; then
    wget https://ultralytics.com/images/zidane.jpg
    wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/yolov8s_pose.hef
else
    echo "Usage: $0 --arch 8    # for Hailo-8"
    echo "       $0 --arch 10   # for Hailo-10"
    exit 1
fi