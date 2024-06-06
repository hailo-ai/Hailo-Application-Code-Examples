Detection and Instance Segmentation example
===========================================

This script performs object detection and instance segmentation on a video using Hailo.
It annotates detected objects with bounding boxes and draws a mask on each instance in every frame of the video.

Requirements
------------

- hailo_platform==4.17.0
- OpenCV (cv2)
- g++-9
 

Usage
-----

1. Install the requierments:
    ```shell script
    sudo apt-get install -y libopencv-dev gcc-9 g++-9
    ```

2. Install PyHailoRT
    - Download the HailoRT whl from the Hailo website - make sure to select the correct Python version. 
    - Install whl:
        ```shell script
        pip install
        ```

3. Download example files:
    ```shell script
    ./download_files.sh
    ```

5. Run the script:
    ```shell script
    python object_detection_tracking.py -m <model_path> -i <input_video_path> -o <output_video_path> -l <label_file_path>
    ```

Arguments
---------

- ``-m, --model``: Path to the pre-trained model file (HEF).
- ``-i, --input``: Path to the input image or video on which object detection will be performed.

Example 
-------
**Command**
```shell script
python supervision_example.py -i input_video.mp4 -o output_video.mp4 -l coco.txt-s 0.5
```

**Input**

![Input example](./input.gif?raw=true)

**Output**

![Output example](./output.gif?raw=true)

Additional Notes
----------------

- The example was only tested with ``HailoRT v4.17.0``
- The example expects a HEF which contains the HailoRT Postprocess
- The script assumes that the input video is in a standard format supported by OpenCV (e.g., .mp4, .avi).

Disclaimer
----------
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.

This example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.

