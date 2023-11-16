**Last HailoRT version checked - 4.15.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.



This is a general YOLO architecture Hailo inference example.  

The example takes one or more images, performs inference using the input HEF file and draws the detection boxes, class type and confidence on the resized image.  
The example works with .jpg, .jpeg, .png and .bmp image files.  

The example was tested with the following Hailo Models Zoo networks:  
yolov3, yolov3_gluon, yolov4_leaky, yolov5s, yolov5m_wo_spp, yolox_l_leaky, yolov6n, yolov7, yolov7_tiny, yolov8m

## Prerequesities:  
numpy  
zenlog  
Pillow  
hailo_platform >= 4.14.0 (installed from the HailoRT .whl, tested on version 4.14.0)  
Hailo Model Zoo prerequesities (tested on versions >=2.8.0)

Install the hailo model-zoo, and hailort whl, and then the requirements:
`pip install -r requiremets.txt`


## Running the example:  
```./yolo_inference.py HEF.hef PATH_TO_IMAGES_FOLDER_OR_IMAGE YOLO_ARCH [--class-num NUM_OF_CLASSES] [--labels LABELS_PATH]```

You can download a sample image and a HEF with the `get_sources.sh` script, and then execute the inference.
for example:  
```CUDA_VISIBLE_DEVICES=9 ./yolo_inference.py ./yolov7.hef ./zidane.jpg yolo_v7```

For more information, run ```./yolo_inference.py --help```   

## IMPORTANT NOTE
As the example, as mentioned above, made to work with COCO trained yolo models, when using a customly trained yolo model, please notice that some values may need to be changed in the relevant functions.  