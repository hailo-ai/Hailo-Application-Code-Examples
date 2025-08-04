#!/usr/bin/bash

if [[ "$1" == "--arch" && "$2" == "8" ]]; then
    wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/images/ocr_img1.png
    wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/images/ocr_img2.png
    wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8/ocr_det.hef
    wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8/ocr.hef
elif [[ "$1" == "--arch" && "$2" == "10" ]]; then
    wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/images/ocr_img1.png
    wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/images/ocr_img2.png
    wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h15/ocr_det.hef
    wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h15/ocr.hef
else
    echo "Usage: $0 --arch 8    # for Hailo-8"
    echo "       $0 --arch 10   # for Hailo-10"
    exit 1
fi