**Last HailoRT version checked - 4.12.0**

This is a HailoRT C++ API yolov5 & yolov7 detection example.

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

`./build/x86_64/vstream_yolov7_example_cpp -hef=YOLO_HEF_FILE.hef -video=VIDEO_FILE.mp4 -arch=ARCH` (where ARCH is yolov5 or yolov7)

NOTE: You can also save the processed video by commenting in a few lines at the "post_processing_all" function in yolov5_yolov7_inference.cpp.

NOTE: There should be no spaces between "=" given in the command line arguments and the file name itself.  
