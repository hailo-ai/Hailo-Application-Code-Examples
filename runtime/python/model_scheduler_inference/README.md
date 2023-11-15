**Last HailoRT version checked - 4.14.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.



This is a general Hailo inference example using the Hailo Model Scheduler.

The example takes one or more images, activates the Model Scheduler (and the Multi-Process Service if specified) and performs inference using the input HEF file.
The example works with .jpg, .png and .bmp image files or with no image files at all.

## Prerequesities:
os  
time  
numpy  
zenlog  
Pillow  
psutil  
multiprocessing  
hailo_platform (installed from the HailoRT .whl of version >= 4.14.0)  

## Running the example:
```./hailo_inference_scheduler.py /path/to/hef/file0 [/path/to/hef/file1 /path/to/hef/file2 ...] [--input-images path/to/image/or/images/folder]  [--use-multi-process]```  

NOTE: This is a very basic example meant for basis of a Python inference code using the Model Scheduler. You are more than welcome to change it to suite your needs.  


For more information, run ```./hailo_inference_scheduler.py -h```
