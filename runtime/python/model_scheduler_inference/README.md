**Last HailoRT version checked - 4.15.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.


This is a general Hailo inference example using the Hailo Model Scheduler.

The example takes one or more images, activates the Model Scheduler (and the Multi-Process Service if specified) and performs inference using the input HEF file.
The example works with .jpg, .png and .bmp image files or with no image files at all.

## Prerequesities: 
zenlog  
Pillow  
psutil    
hailo_platform (installed from the HailoRT .whl of version >= 4.14.0)  

## Running the example:
```./hailo_inference_scheduler.py /path/to/hef/file0 [/path/to/hef/file1 /path/to/hef/file2 ...] [--input-images path/to/image/or/images/folder]  [--use-multi-process]```

You can get example HEF files to test the example with by running ```./get_resource.sh``` and the run:
```./hailo_inference_scheduler.py resnet_v1_50.hef yolov5m_wo_spp_60p_nms_on_hailo.hef yolov8s.hef```

NOTE: This is a very basic example meant for basis of a Python inference code using the Model Scheduler. You are more than welcome to change it to suite your needs.

IMPORTANT NOTE: To be able to run this example, the hailort_serivce must be enabled by running ```sudo systemctl enable hailort.service --now``. Notice that there could be a case where hailort_serivce is enabled, but the inference will fall on "hailort_serivce not enabled". In that case, just run the following command on your terminal:  ```sudo systemctl disable hailort.service --now  && sudo systemctl daemon-reload && sudo systemctl enable hailort.service --now``` and re-run the example.</br>
Some Linux distributions are delivered without the init system ```systemd```, which is required for the multiprocess service. In such a case, it is possible to run the service using the compiled executable ```hailort_service```.

For more information, run ```./hailo_inference_scheduler.py -h```
