## Yolov5 C++ Detection Standalone example for Windows
#### This is an example code for running yolov5 detection using HailoRT and OpenCV on Windows, without TAPPAS dependencies

**Last HailoRT version checked - 4.12.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.



###### prerequisites: Windows 10, HailoRT, OpenCV, [Curl](https://curl.se/windows/)

Tested on Windows 10 Pro, CMake 3.25.1, HailoRT 4.12.0, OpenCV 4.6.0
This example requires as inputs a [yolov5m.hef](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.6.0/yolov5m.hef) and some [traffic video](https://hailo.files.com/files/Hailo_AEs/Examples_Videos). 

Run this script prior to compiling and running yolov5_windows_example in order to get the get those resources and place them where the program looks for them 

```
./get_resources.bat 
```
