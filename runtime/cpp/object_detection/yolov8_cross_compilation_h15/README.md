Yolov8 C++ Object Detection example for H15 
================

This example performs object detection with yolov8 using a hailo15 device.
It receives an input image, hef, frame count and return the image with detected bounding boxes.

Requirements
------------

- hailo_platform==4.18.0
- hailo15 cross development toolchain
- hailo15 platform
- OpenCV 4.5.5
- CMake >= 3.11

Usage
-----
0. Make sure you have the HailoRT correct version and you installed all dependencies. 

1. Download example files:
	```shell script
    ./download_files.sh
    ```

2. Compile the project on the development machine  
	```shell script
    ./build.sh
    ```
	
3. Create a directory in h15 with the project name ( yolov8_example )

4. copy the following files from the build directory in the development machine to hailo15:
	a. yolov8_cross_compilation_h15.
	b. bus.jpg.
	c. yolov8s.hef.

5. Run the example on h15 machine:

	```shell script
    ./yolov8_cross_compilation_h15 -input=<image_path> -model=<hef_path> --frame-count=<number_of_FPS>
    ```
	- this run will infer the bus.jpg frame 30 times , and generate an output.jpg file.
	- the image will have the bounding boxes on it.

6. Copy file and check results:
	now we copy the output.jpg image to our development machine and check to see if the results are what we excpected. 
	
Arguments
---------

- ``-input``: Path to the input image on which object detection will be performed.
- ``--frame-count``: The number of times the network run the image ( frames ) in infer.
- ``-model``: The hef that we want to run on the program.

Example 
-------
**Command**

    ```shell script
	./get_resources.sh
	./build.sh
	`scp build/aarch64/yolov8_cross_compilation_h15 root@10.0.0.1:~/yolov8_example/`
	`scp resources/images/bus.jpg root@10.0.0.1:~/yolov8_example/`
	`scp resources/hefs/h15/yolov8s.hef root@10.0.0.1:~/yolov8_example/`
	`./yolov8_cross_compilation_h15 -input=bus.jpg -hef=yolov8s_h15.hef -num=30`.
	```	



**Output**

[output_example] ( output.jpg ) 

Notes
----------------
- The example was only tested with ``HailoRT v4.18.0``
- The script assumes that the image is in one of the following formats: .jpg, .jpeg, .png or .bmp 
- There should be no spaces between "=" given in the command line arguments and the file name itself.

Disclaimer
----------
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.

This example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.
