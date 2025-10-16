#!/usr/bin/bash

if [[ "$1" == "--arch" && "$2" == "8" ]]; then
    wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/images/input_image.png
    wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8/real_esrgan_x2.hef

elif [[ "$1" == "--arch" && "$2" == "10" ]]; then
    wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/images/input_image.png
    wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo15h/real_esrgan_x2.hef
else
    echo "Usage: $0 --arch 8    # for Hailo-8"
    echo "       $0 --arch 10   # for Hailo-10"
    exit 1
fi