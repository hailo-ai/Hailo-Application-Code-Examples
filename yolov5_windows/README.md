## Yolov5 C++ Detection Standalone example for Windows
#### This is an example code for running yolov5 detection using HailoRT and OpenCV on Windows, without TAPPAS dependencies

###### prerequisites: Windows 10, HailoRT, OpenCV, [Curl](https://curl.se/windows/)

Tested on Windows 10 Pro, CMake 3.25.1, HailoRT 4.12.0, OpenCV 4.6.0
This example requires as inputs a [yolov5m.hef](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.6.0/yolov5m.hef) and some [traffic video](https://hailo.files.com/files/Hailo_AEs/Examples_Videos). 

Run this script prior to compiling and running yolov5_windows_example in order to get the get those resources and place them where the program looks for them 

```
./get_resources.bat 
```
