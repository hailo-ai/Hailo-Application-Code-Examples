
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
