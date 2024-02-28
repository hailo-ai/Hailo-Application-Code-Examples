**Last HailoRT version checked - 4.16.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.


This is a HailoRT C++ API yolov5 & yolov7 detection example. Note that this example also supports yolov5\v7 models with NMS on-Hailo. 

The example does the following:

1. Creates a device (pcie)
2. Reads the network configuration from a yolov5\yolov7 HEF file
3. Prepares the application for inference
4. Runs inference and postprocess on a given video file 
5. Draws the detection boxes on the original video
6. Prints the object detected + confidence to the screen
5. Prints statistics

NOTE: Currently supports only devices connected on a PCIe link.

Prequisites:
OpenCV 4.2.X
CMake >= 3.20
HailoRT >= 4.10.0
git - to clone the rapidjson repository

To compile the example run `./build.sh`

To run the compiled example:

`./build/x86_64/vstream_yolov5_yolov7_example_cpp -hef=YOLO_HEF_FILE.hef -input=VIDEO_FILE.mp4 -arch=ARCH` (where ARCH is yolov5 or yolov7)

For example:
`./build/x86_64/vstream_yolov5_yolov7_example_cpp -hef=yolov5m_wo_spp.hef -input=full_mov_slow.mp4 -arch=yolov5`

NOTE: When using a HEF file that was compiled with NMS on-Hailo, the `-arch` is redundant. For the regular compiled model, it is mandatory. 

NOTE: You can also save the processed video by commenting in a few lines at the "post_processing_all" function in yolov5_yolov7_inference.cpp.

NOTE: There should be no spaces between "=" given in the command line arguments and the file name itself.  
