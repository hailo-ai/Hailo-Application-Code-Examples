#!/bin/bash

# This script downloads the necessary resources for Hailo demos
# from AWS and places them in the designtad folders

# Define the URL for the video files and the hef file
VIDEO_URL="https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/"
HEF_URL="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.6.1/"

# Define the names of the video files and the hef file
VIDEO_FILES=(
   "detection0.mp4"
   "detection1.mp4"
   "detection2.mp4"
   "detection3.mp4"
   "detection4.mp4"
   "detection5.mp4"
   "detection6.mp4"
   "detection7.mp4"
   "detection8.mp4"
   "detection9.mp4"
   "detection10.mp4"
   "detection11.mp4"
   "detection12.mp4"
    "detection13.mp4"
    "detection14.mp4"
)

HEF_FILE="yolox_s_leaky.hef"

# Create the resources/hefs and resources/video directories if they do not exist
if [ ! -d "resources/hefs" ]; then
    mkdir -p resources/hefs
fi

if [ ! -d "resources/video" ]; then
    mkdir -p resources/video
fi

# Download the video files to the resources/video directory
for file in "${VIDEO_FILES[@]}"; do
    wget --directory-prefix=resources/video "${VIDEO_URL}${file}"
    if [ "$?" -ne 0 ]; then
        echo "Error: Download failed for ${file}" >&2
        exit 1
    fi
done

# Download the hef file to the resources/hefs directory
wget --directory-prefix=resources/hefs "${HEF_URL}${HEF_FILE}"
if [ "$?" -ne 0 ]; then
    echo "Error: Download failed for ${HEF_FILE}" >&2
    exit 1
fi

echo "Downloads successful"

