#!/bin/bash

# This script downloads the necessary resources for Hailo demos
# from AWS and places them in the designtad folders

# Define the URL for the video files and the hef file
IMAGE_URL="https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/images/"
HEF_URL="https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h15/"
PLATFORM="h15"
HEF_FILE="yolov8s.hef"
IMAGE_FILE="bus.jpg"

# Create the resources/hefs and resources/video directories if they do not exist
if [ ! -d "resources/hefs" ]; then
    mkdir -p resources/hefs
fi

if [ ! -d "resources/images" ]; then
    mkdir -p resources/images
fi

# Download the video files to the resources/video directory
    wget --directory-prefix=resources/images/ "${IMAGE_URL}${IMAGE_FILE}"
    if [ "$?" -ne 0 ]; then
        echo "Error: Download failed for ${file}" >&2
        exit 1
    fi

# Download the hef file to the resources/hefs directory
wget --directory-prefix=resources/hefs/h15/ "${HEF_URL}${HEF_FILE}"
if [ "$?" -ne 0 ]; then
    echo "Error: Download failed for ${HEF_FILE}" >&2
    exit 1
fi

echo "Downloads successful"

