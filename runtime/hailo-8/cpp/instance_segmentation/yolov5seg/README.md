Detection and Instance Segmentation example
===========================================

This script performs object detection and instance segmentation on a video using Hailo.
It annotates detected objects with bounding boxes and draws a mask on each instance in every frame of the video.

Requirements
------------

- hailo_platform==4.18.0
- OpenCV (cv2)
- g++-9
 

Usage
-----

1. Install the requirements:
    ```shell script
    sudo apt-get install -y libopencv-dev gcc-9 g++-9
    ```

2. Download example files:
    ```shell script
    ./get_hef_and_video.sh
    ```

3. Compile the example:
    `./build.sh`

4. Run the script:
    `./build/yolov5seg_example_cpp --model=yolov5m-seg.hef --input=full_mov_slow.mp4`

Arguments
---------

- ``-m, --model``: Path to the pre-trained model file (HEF).
- ``-i, --input``: Path to the input image or video on which object detection will be performed.


Output
------
![output example](./processed_video.gif)

Additional Notes
----------------

- The example was only tested with ``HailoRT v4.18.0``
- The example expects a HEF which contains the HailoRT Postprocess
- The script assumes that the input video is in a standard format supported by OpenCV (e.g., .mp4, .avi).

Disclaimer
----------
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.

This example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.

