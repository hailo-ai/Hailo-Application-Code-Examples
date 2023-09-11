Raw Streams Async Example (yolov5)  

This example demonstrates the use of HailoRT async API.

To get hef and video:  
``
get_hef_and_video.sh
``

To build:  
``
./build.sh
``

To run:  
``
./build/async_infer
vlc processed_video.mp4
``

What this example does?  
This example creates 

Notes:  
If you want to save the processed video as mp4, search in this code this note: ```// add in order to save to file the processed video```
and add it (there are 3 lines to un-note).  