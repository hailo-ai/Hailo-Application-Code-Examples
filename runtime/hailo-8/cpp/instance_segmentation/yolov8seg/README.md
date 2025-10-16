**Last HailoRT version checked - 4.18.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.


This is a hailort C++ API yolov8_seg instance segmentation example.

The example does the following:

1. Creates a device (pcie)
2. Reads the network configuration from a yolov8_pose HEF file
3. Prepares the application for inference
4. Runs inference and postprocess (using xtensor library) on a given image\video file or a camera 
5. Draws the detection boxes and masking on image\video\camera input (self defined by the user)
6. Prints the object detected + confidence to the screen
7. Prints statistics

**NOTE**: Currently support only devices connected on a PCIe link.


Prequisites:

OpenCV 4.2.X

CMake >= 3.20

HailoRT >= 4.10.0

Xtensor - no installation or build needed as it's complied from the web as an external project, but git is required.


**IMPORTANT NOTE**: You need to set the BASE_DIR variable in the CMakeLists.txt to be the folder path to the location of the yolov8_pose_cpp folder.



To compile the example run `./build.sh`

To run the compiled example:

For an image:
`./build/x86_64/vstream_yolov8seg_example_cpp -hef=YOLOv8seg_HEF_FILE.hef -input=IMAGE_FILE.jpg [-num=NUM_TIMES]`
For a video:
`./build/x86_64/vstream_yolov8seg_example_cpp -hef=YOLOv8seg_HEF_FILE.hef -input=VIDEO_FILE.mp4`
For a camera input:
`./build/x86_64/vstream_yolov8seg_example_cpp -hef=YOLOv8seg_HEF_FILE.hef -input=`

Example:
`./build/x86_64/vstream_yolov8seg_example_cpp -hef=yolov8s_seg.hef -input=full_mov_slow.mp4`


**NOTE**: This example uses xtensor C++ ibrary compiled from the xtl git as an external source. 

**NOTE**: You can also save the processed image\video by commenting in a few lines in the "post_processing_all" function - for images, the cv::imwrite line and for a video the other commented out lines.

**NOTE**: There should be no spaces between "=" given in the command line arguments and the file name itself.

**NOTE**: You can, and sometimes need to, change the values of NUM_CLASSES, IOU_THRESHOLD and SCORE_THRESHOLD in the yolov8seg_postprocess.cpp file for different videos and different compiled yolov8seg models.

**NOTE**: In case you run the example with a single image, the "-num=NUM_TIMES" flag will instruct the application how many times to run the same image. This is used to measure the performance without the overhead of ecoding a video file. In case you use a video (.avi or .mp4 file), the "-num" flag will have no effect.
