**Last HailoRT version checked - 4.16.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.


This is a Yolo8-pose Hailo inference example.  

The example takes one image or a folder containing images, performs inference using the input HEF file and draws the detection boxes, class type, confidence, keypoints and joints connection on the resized image.  
The example works with .jpg, .jpeg, .png and .bmp image files.  

The example was tested with the following Hailo Models Zoo networks:  
yolov8s-pose

## Prerequesities:
numpy  
zenlog  
Pillow  
hailo_platform >= 4.14.0 (installed from the HailoRT .whl)  
Hailo Model Zoo prerequesities (tested on versions >=2.8.0)

Install the hailo model-zoo, and hailort whl, and then the requirements:
`pip install -r requiremets.txt`


## Running the example:  
```./yoloseg_inference.py HEF.hef PATH_TO_IMAGES_FOLDER_OR_IMAGE YOLOSEG_ARCH [--class-num NUM_OF_CLASSES]```

You can download sample images and a HEF with the `get_sources.sh` script, and then execute the inference.
For example:  
```./yolov8_pose_inference.py yolov8s_pose_mz.hef .```

For more information, run ```./yolov8_pose_inference.py --help```   

