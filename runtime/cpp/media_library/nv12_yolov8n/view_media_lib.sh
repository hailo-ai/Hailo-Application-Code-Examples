#!/usr/bin/bash
gst-launch-1.0 -v udpsrc port=5000 address=10.0.0.2 ! application/x-rtp,encoding-name=H264 ! queue ! rtph264depay ! queue ! h264parse ! avdec_h264 ! queue ! videoconvert ! videocrop bottom=8 ! fpsdisplaysink text-overlay=false sync=false
