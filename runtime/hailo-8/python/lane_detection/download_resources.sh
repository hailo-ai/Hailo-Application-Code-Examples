#!/usr/bin/bash

if [[ "$1" == "--arch" && "$2" == "8" ]]; then
    wget https://www.pexels.com/download/video/4608277/?fps=24.0&h=720&w=1280 -O input_video.mp4
    wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8/ufld_v2_tu.hef
elif [[ "$1" == "--arch" && "$2" == "10" ]]; then
    wget https://www.pexels.com/download/video/4608277/?fps=24.0&h=720&w=1280 -O input_video.mp4
    wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h15/ufld_v2_tu.hef
else
    echo "Usage: $0 --arch 8    # for Hailo-8"
    echo "       $0 --arch 10   # for Hailo-10"
    exit 1
fi