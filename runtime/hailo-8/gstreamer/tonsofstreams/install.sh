#!/usr/bin/bash
#!/bin/bash

# This script downloads the necessary resources for Hailo demos
# from AWS and places them in the designtad folders

# Create the resources/hefs and resources/video directories if they do not exist
if [ ! -d "resources/hefs" ]; then
    mkdir -p resources/hefs
fi
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.6.1/yolov5m_wo_spp_60p.hef -O resources/hefs/yolov5m_wo_spp_60p.hef
if [ "$?" -ne 0 ]; then
    echo "Error: Download failed for yolov5m.hef" >&2
    exit 1
fi

if [ ! -d "resources/video/bytestreams" ]; then
    mkdir -p resources/video/bytestreams
fi


# Download the bytestreams library
for i in {0..49}
do
    wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/bytestreams/bytestream$i.mp4 -O resources/video/bytestreams/bytestream$i.mp4
    if [ "$?" -ne 0 ]; then
        echo "Error: Download failed for bytestream$i.mp4" >&2
        exit 1
    fi
done

echo "Downloads successful"