#!/bin/bash

# download hef files to ./resources
#H8 HEFs
wget -nc https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.9.0/clip_resnet_50x4.hef -P ./resources
wget -nc https://hailo-tappas.s3.eu-west-2.amazonaws.com/v3.26/general/hefs/yolov5s_personface.hef  -P ./resources

#H8L (RPi) HEFs
wget -nc https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8l_rpi/clip_resnet_50x4_h8l.hef -P ./resources
wget -nc https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8l_rpi/yolov5s_personface_h8l_pi.hef -P ./resources

wget -nc https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/clip_example.mp4 -P ./resources