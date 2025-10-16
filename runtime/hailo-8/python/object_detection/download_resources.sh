#!/usr/bin/bash

if [[ "$1" == "--arch" && "$2" == "8" ]]; then
    wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/images/bus.jpg
    wget https://hailo-tappas.s3.eu-west-2.amazonaws.com/v3.22/general/media/full_mov_slow.mp4
    wget -q https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8n.hef
elif [[ "$1" == "--arch" && "$2" == "10" ]]; then
    wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/images/bus.jpg
    wget https://hailo-tappas.s3.eu-west-2.amazonaws.com/v3.22/general/media/full_mov_slow.mp4
    wget -q https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/yolov8n.hef
else
    echo "Usage: $0 --arch 8    # for Hailo-8"
    echo "       $0 --arch 10   # for Hailo-10"
    exit 1
fi