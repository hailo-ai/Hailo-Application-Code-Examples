Detection C++ Object Detection example for Hailo8
================

This example performs object detection with Yolo detection & MobileNet-SSD models using a hailo8 device.
It receives an image\video\camera as input, hef, and optional frame count and either prints the detected object + score to the screen, or returns the image\video with detected bounding boxes.

Requirements
------------

- hailo_platform==4.17.0
- OpenCV >= 4.2.X
- CMake >= 3.20

Usage
-----
0. Make sure you have the HailoRT correct version and you installed all dependencies. 

1. Download example files:
	```shell script
    ./build.sh
    ```
    Which copy the following files to the example's directory:
    fsd

2. Compile the project on the development machine  
	```shell script
    ./build.sh
    ```
	Which creates the directory hierarchy build/x86 and compile an executable file called detection_example_cpp

5. Run the example:

	```shell script
    ./vstream_detection_example_cpp -hef=<hef_path> -input=<image_or_video_or_camera_path> -num=<number_of_times_to_run_one_image>
    ```
 
	
Arguments
---------

- ``-input``: Path to the input image\video\camera on which object detection will be performed.
- ``-hef``: Path to HEF file to run inference on.
- ``-num (optional)``: Number of times to run smae image. Only relevant when running the example with a single image.

Example 
-------
**Command**

    ```shell script
	./get_resources.sh
	./build.sh
    For video:
	`./vstream_detection_example_cpp -hef=yolov7.hef -input=full_mov_slow.mp4`
    For a single image:
    `./vstream_detection_example_cpp -hef=yolov8s.hef -input=bus.jpg`
    For a single image multiple times:
    `./vstream_detection_example_cpp -hef=yolox_tiny.hef -input=bus.jpg -num=200`
	```	

Notes
----------------
- Last HailoRT version checked - ``HailoRT v4.17.0``
- The script assumes that the image is in one of the following formats: .jpg, .jpeg, .png or .bmp 
- There should be no spaces between "=" given in the command line arguments and the file name itself
- The example only works for detection models that have the NMS on-Hailo (either on the NN-core or on the CPU)

Disclaimer
----------
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.

This example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.
