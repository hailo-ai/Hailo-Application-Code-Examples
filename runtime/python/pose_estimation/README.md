**Last HailoRT version checked - 4.17.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.


This is a Yolo8-pose Hailo inference example.  

The example takes one image or a folder containing images, performs inference using the input HEF file and draws the detection boxes, class type, confidence, keypoints and joints connection on the resized image.  
The example works with .jpg, .jpeg, .png and .bmp image files.  

## Prerequesities:

### HailoRT python package:
hailo_platform >= 4.17.0 (installed from the HailoRT .whl)  
For example:
`pip install hailort-4.17.0-cp310-cp310-linux_x86_64.whl`

### Dependencies:
zenlog  
pillow  
opencv-python

(numpy as part of hailo_platform dependencies)
To install the requirements:
`pip install -r requirements.txt`



## Running the example:  
The template for running the example is:
```./yolov8_pose_inference.py <HEF> <PATH_TO_IMAGES_FOLDER_OR_IMAGE>```

The annotated files will be saved in the `output_images` folder.

To run with H8 run for example:
`./yolov8_pose_inference.py yolov8s_pose_mz.hef person.jpg`
To run with H8L run for example:
`./yolov8_pose_inference.py yolov8s_pose_h8l_pi.hef zidane.jpg`

You can download sample images and a HEF with the `get_sources.sh` script, and then execute the inference.
For example:  
```./yolov8_pose_inference.py yolov8s_pose_mz.hef .```

For more information, run ```./yolov8_pose_inference.py --help```   

