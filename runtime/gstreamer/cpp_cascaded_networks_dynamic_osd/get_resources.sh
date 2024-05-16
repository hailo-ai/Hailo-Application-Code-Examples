#!/bin/bash

# Download the files and place them in the resources folder
echo "Downloading resources..."

wget -P resources https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h15/yolov8s_vga_nv12.hef
wget -P resources https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h15/yolov5s_hd_nv12.hef
wget -P resources https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h15/mspn_regnetx_800mf_nv12.hef

echo "Done"
