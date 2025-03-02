Object Detection
================
This example performs object detection using a Hailo8 device.
It receives a HEF and images/video/camera as input, and returns the image\video with annotations of detected objects and bounding boxes.

![output example](./obj_det.gif)

Requirements
------------

- hailo_platform==4.20.0
- OpenCV >= 4.5.4
- CMake >= 3.20

Supported Models
----------------
This example expects the HEF to contain HailoRT-Postprocess. 

Because of that, this example only supports detections models that allow HailoRT-Postprocess:
- yolov5/6/7/8/9/10/11
- yolox
- ssd
- centernet


Usage
-----
0. Make sure you have the HailoRT correct version and you installed all dependencies. 

1. Clone the repository:
    ```shell script
    git clone <https://github.com/hailo-ai/Hailo-Application-Code-Examples.git>
        
    cd Hailo-Application-Code-Examples/runtime/cpp/object_detection
    ``` 

2. Download example files:
	```shell script
    ./get_resources.sh
    ```
    Which copies the following files to the example's directory:
    ```
    full_mov_slow.mp4
    bus.jpg
    yolov8n.hef
    ```

3. Compile the project on the development machine  
	```shell script
    ./build.sh
    ```
	Which creates the directory hierarchy build/x86 and compile an executable file called obj_det

5. Run the example:

	```shell script
    ./build/x86_64/obj_det -hef=<hef_path> -input=<image_or_video_or_camera_path>
    ```
	
Arguments
---------

- ``-input``: Path to the input image\video\camera on which object detection will be performed.
- ``-hef``: Path to HEF file to run inference on.
- ``-s (optional)``: A flag for saving the output video of a camera input. 

Example 
-------
**Command**

	./get_resources.sh

	./build.sh

    For video:
	./build/x86_64/obj_det -hef=yolov8n.hef -input=full_mov_slow.mp4
    For a single image:
    ./build/x86_64/obj_det -hef=yolov8s.hef -input=bus.jpg
    For a directory of images:
    ./build/x86_64/obj_det -hef=yolov8s.hef -input=images
    For saving camera input:
    ./build/x86_64/obj_det -hef=yolox_tiny.hef -input=/dev/video0 -s

Notes
----------------
- Last HailoRT version checked - ``HailoRT v4.20.0``
- The script assumes that the image is in one of the following formats: .jpg, .jpeg, .png or .bmp 
- There should be no spaces between "=" given in the command line arguments and the file name itself
- The example only works for detection models that have the NMS on-Hailo (either on the NN-core or on the CPU)
- When using camera as input:
    - To exit gracefully from openCV window, press 'q'.
    - Camera path is usually found under /dev/video0 .
    - In case OpenCV is defaulting to GStreamer for video capture warnings might occur.
      To solve, force OpenCV to use V4L2 instead of GStreamer by setting these environment variables:
      ```
        export OPENCV_VIDEOIO_PRIORITY_GSTREAMER=0
        export OPENCV_VIDEOIO_PRIORITY_V4L2=100
      ```

    
    

Disclaimer
----------
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.

This example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.
