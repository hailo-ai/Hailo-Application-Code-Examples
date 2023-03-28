This example runs multiple streams on multiple Hailo devices.

It is built to work with 4 Hailo devices.
To download required HEF and media files run ./install.sh script.
Note that the mp4 files used here are not regular mp4 files. They are modified to be used as streaming media files.
This is done to allow to run them in loop. see scripts/gstreamer/gstreamer_video_converter.sh for details

Requirements:
- TAPPAS environment (tested on TAPPAS 3.24.0)
- 4 Hailo devices

To run the example run ./tonsofstreams.sh
