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

NOTE: This is a very basic example meant for basis of a Python inference code using the Model Scheduler. You are more then welcome to change it to suite your needs.


For more information, run ```./hailo_inference_scheduler.py -h```
