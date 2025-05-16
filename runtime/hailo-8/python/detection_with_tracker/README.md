Object Detection with Tracking using ByteTracker and Supervision
================================================================

This script performs object detection with tracking on a video using Hailo.
It annotates detected objects with bounding boxes and labels and tracks them across frames in the video.

![Output example](./output.gif?raw=true)

Requirements
------------

- hailo_platform==4.18.0
- Roboflow supervision
- OpenCV (cv2)
- tqdm

Note: Roboflow supervision ships ByteTracker already. Thus, you needn't install bytetracker

Supported Models
----------------

This example expects the HEF model to contain HailoRT-Postprocess. Because of that, this example only supports detections models that allow hailort-postprocess:
- yolov5/6/7/8
- yolox
- ssd
- centernet
 
Usage
-----

0. Create a virtualenv
    ```shell script
    python -m venv .venv
    source .venv/bin/activate
    ```

1. Install PyHailoRT
    - Download the HailoRT whl from the Hailo website - make sure to select the correct Python version. 
    - Install whl:
        ```shell script
        pip install hailort-X.X.X-cpXX-cpXX-linux_x86_64.whl
        ```

2. Clone the repository:
    ```shell script
    git clone https://github.com/hailo-ai/Hailo-Application-Code-Examples.git
        
    cd Hailo-Application-Code-Examples/runtime/hailo-8/python/detection_with_tracker/
    ```

3. Install dependencies:
    ```shell script
    pip install -r requirements.txt
    ```

4. Download example files:
    ```shell script
    ./download_resources.sh
    ```

5. Run the script:
    ```shell script
    ./object_detection_tracking.py -n <net> -i <input_video> -o <output_video> -l <labels> -s <score_thresh>
    ```

As a result, the program will perform the detection/tracking of objects in the video file, generating the output video in the same directory:

![image](https://github.com/user-attachments/assets/1b7fe0ab-eb73-4f47-b493-ec17832931a9)

Arguments
---------

- ``-n, --net``: Path to the pre-trained model file (HEF).
- ``-i, --input_video``: Path to the input video on which object detection will be performed.
- ``-o, --output_video``: Path to save the output video with annotated objects.
- ``-l, --labels``: Path to a text file containing class labels for the detected objects.
- ``-s, --score-thresh``: Score threshold used for discarding detections.

For more information:
```shell script
./detection_with_tracker.py -h
```
Example 
-------
**Command**
```shell script
./detection_with_tracker.py -n yolov5m_wo_spp_60p.hef -i input_video.mp4 -o output_video.mp4 -l coco.txt -s 0.5
```

Additional Notes
----------------

- The example was only tested with `HailoRT v4.18.0` and `HailoRT v4.21.0`
- The example expects a HEF model file which contains the HailoRT Postprocess
- The script assumes that the input video is in a standard format supported by OpenCV (e.g., .mp4, .avi).

Disclaimer
----------
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.

This example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.

