#!/usr/bin/bash
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.4.0/yolov5s_personface.hef
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.4.0/repvgg_a0_person_reid_2048.hef
wget https://hailo-tappas.s3.eu-west-2.amazonaws.com/v3.21/general/media/re_id/reid.tar.gz
mkdir video_images
ffmpeg -i reid0.mp4 -vf "fps=30" video_images/image%d.png
