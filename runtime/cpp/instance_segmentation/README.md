**Last HailoRT version checked - 4.16.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.


This is a HailoRT C++ API yolov5seg detection + instance segmentation example.

The example does the following:

Creates a device (pcie)
Reads the network configuration from a yolov5eg HEF file
Prepares the application for inference
Runs inference and postprocess on a given video file
Draws the detection boxes on the original video
Colors the detected objects pixels
Prints the object detected + confidence to the screen
Prints statistics
NOTE: Currently supports only devices connected on a PCIe link.

Prequisites: OpenCV 4.2.X CMake >= 3.20 HailoRT >= 4.10.0 git - rapidjson repository is cloned when performing build.

To compile the example run `./build.sh`

To run the compiled example:


`./build/vstream_yolov5seg_example_cpp -hef=yolov5n_seg.hef -input=full_mov_slow.mp4`

NOTE: You can also save the processed video by commenting in a few lines at the `post_processing_all` function in yolov5seg_example.cpp.

NOTE: There should be no spaces between "=" given in the command line arguments and the file name itself.
