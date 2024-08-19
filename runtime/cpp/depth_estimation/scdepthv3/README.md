**Last HailoRT version checked - 4.18.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.


SCDepthV3 Inference Example
---------------------------

This example uses the C++ API of HailoRT to implement a depth estimation inference example. The postprocess is suitable for the scdepth architecture.

The inputs are a compiled network (HEF) file and
a video file (.mp4 or .avi)
The output is a processed depthmap video file (output_video.mp4)

This example was tested on this setup:
- x86 machine
- M-key Hailo-8
- HailoRT 4.14.0/4.15.0 


1. Dependencies:
    - OpenCV, and g++-9:
    ``` bash
    sudo apt-get install -y libopencv-dev gcc-9 g++-9
    ```
2. Download files:
    ``` bash
    ./download_files.sh
    ```
    This will download the scdepthv3 hef and a video example input_video.mp4.
3. Build the project 
    ``` bash
    ./build.sh
    ```
4. Run the executable:
    ``` bash
    ./build/depth_estimation_example_cpp -hef=HEF_PATH -path=VIDEO_PATH
    ```
    - To use the downloaded files, run:
        ``` bash
        ./build/depth_estimation_example_cpp -hef=scdepthv3.hef -path=input_video.mp4
        ```
The output processed video is saved as **output_video.mp4**
