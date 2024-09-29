Object Detection
================

This example performs object detection using a Hailo8 device.
It receives an input image and annotates it with detected objects and bounding boxes.

![output example](./output_image0.jpg)

Requirements
------------

- hailo_platform==4.18.0
- Pillow
- numpy
- loguru

Supported Models
----------------

This example expects the hef to contain HailoRT-Postprocess. 

Because of that, this example only supports detections models that allow hailort-postprocess:
- yolov5/6/7/8
- yolox
- ssd
- centernet
 

Usage
-----

0. Install PyHailoRT
    - Download the HailoRT whl from the Hailo website - make sure to select the correct Python version. 
    - Install whl:
        ```shell script
        pip install hailort-X.X.X-cpXX-cpXX-linux_x86_64.whl
        ```

1. Clone the repository:
    ```shell script
    git clone <https://github.com/hailo-ai/Hailo-Application-Code-Examples.git>
        
    cd object_detection
    ```

2. Install dependencies:
    ```shell script
    pip install -r requirements.txt
    ```

3. Download example files:
    ```shell script
    ./download_resources.sh
    ```

4. Run the script:
    ```shell script
    ./object_detection -n <model_path> -i <input_image_path> -o <output_path> -l <label_file_path>
    ```

Arguments
---------

- ``-n, --net``: Path to the pre-trained model file (HEF).
- ``-i, --input``: Path to the input image on which object detection will be performed.
- ``-o, --output``: Path to save the output image with annotated objects.
- ``-l, --labels``: Path to a text file containing class labels for the detected objects.

For more information:
```shell script
./object_detection.py -h
```
Example 
-------
**Command**
```shell script
./object_detection.py -n ./yolov7.hef -i zidane.jpg
```

Additional Notes
----------------

- The example was only tested with ``HailoRT v4.18.0``
- The example expects a HEF which contains the HailoRT Postprocess
- The script assumes that the image is in one of the following formats: .jpg, .jpeg, .png or .bmp 

Disclaimer
----------
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.

This example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.
