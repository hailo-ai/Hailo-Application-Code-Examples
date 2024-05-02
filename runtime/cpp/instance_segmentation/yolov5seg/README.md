**Last HailoRT version checked - 4.17.0**

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

1. Dependencies:
    - OpenCV, and g++-9:
    ``` bash
    sudo apt-get install -y libopencv-dev gcc-9 g++-9
    ```
2. Build the project build.sh
3. Run the executable:
    ``` bash
    ./build/segmentation_example_cpp -hef=HEF_PATH -path=VIDEO_PATH
    ```
    This example contains the hef file yolov5n-seg.hef and the video full_mov_slow.mp4, so for a quick demo you can run:
    ``` bash
    ./build/segmentation_example_cpp -hef=yolov5n-seg.hef -path=full_mov_slow.mp4

The output is a processed segmented video file - 'processed_video.mp4'

NOTE: There should be no spaces between "=" given in the command line arguments and the file name itself.
