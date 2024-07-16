#!/usr/bin/bash
wget https://ultralytics.com/images/zidane.jpg
ffmpeg -i zidane.jpg -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2" zidane_FHD.jpg
rm zidane.jpg

