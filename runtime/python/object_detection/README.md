Object Detection
================

This example performs object detection using a Hailo8 device.
It processes input images, videos, or a camera stream and annotates it with the detected objects.

![output example](./output.gif)

Requirements
------------

- hailo_platform==4.20.0
- opencv
- numpy
- loguru

Supported Models
----------------

This example only supports detections models that allow HailoRT-Postprocess:
- YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10
- YOLOx
- SSD
- CenterNet
 

Usage
-----
To avoid compatibility issues, it's recommended to have a separate venv from the DFC.

0. Install PCIe driver and PyHailoRT
    - Download and install the PCIe driver and PyHailoRT from the Hailo website
    - To install the PyHailoRT whl:
        ```shell script
        pip install hailort-X.X.X-cpXX-cpXX-linux_x86_64.whl
        ```

1. Clone the repository:
    ```shell script
    git clone <https://github.com/hailo-ai/Hailo-Application-Code-Examples.git>
        
    cd Hailo-Application-Code-Examples/runtime/python/object_detection
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
    ./object_detection -n <model_path> -i <input_image_path> -l <label_file_path> -b <batch_size>
    ```
The output results will be saved under a folder named output.

Arguments
---------

- ``-n, --net``: Path to the pre-trained model file (HEF).
- ``-i, --input``: Path to the input image on which object detection will be performed.
- ``-l, --labels``:[optional] Path to a text file containing class labels for the detected objects.
- ``-s, --save_stream_output``:[optional] Save the output of the inference from a stream.
- ``-b, --batch_size``:[optional] Number of images in one batch. Defaults to 1.

For more information:
```shell script
./object_detection.py -h
```
Example 
-------
**Inference on a camera stream**
```shell script
./object_detection.py -n ./yolov7.hef -i camera
```
**Inference on a video**
```shell script
./object_detection.py -n ./yolov7.hef -i input_video.mp4
```
**Inference on an image**
```shell script
./object_detection.py -n ./yolov7.hef -i zidane.jpg
```
**Inference on a folder of images**
```shell script
./object_detection.py -n ./yolov7.hef -i input_folder
```

Additional Notes
----------------

- The example was only tested with ``HailoRT v4.20.0``
- The example expects a HEF which contains the HailoRT Postprocess
- Images are only supported in the following formats: .jpg, .jpeg, .png or .bmp
- Number of input images should be divisible by batch_size
- For any issues, open a post on the [Hailo Community](https://community.hailo.ai)

Disclaimer
----------
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.

This example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.
