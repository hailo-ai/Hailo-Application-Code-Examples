## Yolox Python Streaming Inference
#### This is an example code for running yolovx detection using HailoRT and OpenCV. It's using Hailo Model Zoo module for postprocessing.

###### prerequisites: HailoRT, OpenCV, wget, [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo), Camera (required only for streaming from camera)

Tested on Ubuntu 20.04, HailoRT 4.12.1, OpenCV 4.6.0
This example requires yolox_s_leaky.hef and video resources in order to run. 

Usage:
- Run the get_resources script once prior to running the example. 
- Run the example.
- When the example is running, Use the keyboard to toggle between 
- - camera (`c`) (if available) 
- - video (`v`) (selects a random video)
- - quit (`q`) to quit the example


```
./get_resources.sh 
python3 ./yolox_stream_inference.py
```
