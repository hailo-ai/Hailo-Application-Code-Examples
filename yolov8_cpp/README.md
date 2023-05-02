** Last HailoRT version checked - 4.12.0 **
This is a hailort C++ API yolov8 detection example.

The example does the following:

1. Creates a device (pcie)
2. Reads the network configuration from a yolov8 HEF file
3. Prepares the application for inference
4. Runs inference and postprocess (using xtensor library) on a given image\video file 
5. Draws the detection boxes on the original image\video
6. Prints the object detected + confidence to the screen
5. Prints statistics

NOTE: Currently support only devices connected on a PCIe link.
NOTE: This example was tested with a yolov8m model.

Prequisites:
OpenCV 4.2.X
CMake >= 3.20
HailoRT >= 4.10.0
Xtensor


To compile the example run `./build.sh`

To run the compiled example:

For an image:
`./build/x86_64/vstream_yolov8_example_cpp -hef=YOLOv8_HEF_FILE.hef -input=IMAGE_FILE.jpg`
For a video:
`./build/x86_64/vstream_yolov8_example_cpp -hef=YOLOv8_HEF_FILE.hef -input=VIDEO_FILE.mp4`

NOTE: This example uses xtensor C++ ibrary compiled from the xtl git as an external source. 

NOTE: You can also save the processed image\video by commenting in a few lines in the "post_processing_all" function - for images, the cv::imwrite line and for a video the other commented out lines.

NOTE: There should be no spaces between "=" given in the command line arguments and the file name itself.

NOTE: You can play with the values of IOU_THRESHOLD and SCORE_THRESHOLD in the yolov8_postprocess.cpp file for different videos to get more detections. 

NOTE: In case you prefer to perform the Sigmoid on host, you can comment in the relevant line to do that. Please notice that you'll need a HEF file that does not have an on-chip sigmoid if you choose to use the example in such a way. 

IMPORTANT NOTE: The pre-compiled Yolov8 HEF files Hailo Model Zoo are compiled with normalization and sigmoid activation on-chip and have a 16-bit output layer.
In the example we assume that the the *entire model is 8-bit*.
Both scores and data dequantization is done manually in the postprocessing functions. 
This means that you will not get good detection with this example using the Yolov8 Hailo Model Zoo HEFs. 
To perform inference & postprocess for the Yolov8 HEFs in Hailo Model Zoo, you can use the Hailo Model Zoo eval function. 
