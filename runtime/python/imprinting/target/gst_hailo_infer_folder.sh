#!/bin/bash

rm output.txt

if [ $# -gt 0 ]; then
  FOLDER="$1"
else
  FOLDER="/home/root/pocs/online_training/training_data/002.Laysan_Albatross" 
fi

PIPELINE="gst-launch-1.0 \
          multifilesrc location=$FOLDER/%04d.jpg \
          ! jpegdec ! videoscale ! video/x-raw,width=224,height=224 ! videoconvert \
          ! hailonet hef-path=/home/root/pocs/online_training/hef/resnet_v1_18_featext.hef \
          ! hailofilter function-name=filter \
               so-path=/home/root/pocs/online_training/target/libfeature_extractor_cross_compilation_h15.so 
	  ! filesink location=stam.txt"


eval ${PIPELINE}

cp "output.txt" $FOLDER/

