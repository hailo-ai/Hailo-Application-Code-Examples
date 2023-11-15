## Yolox Python Streaming Inference
#### This is an example code for running yolovx detection using HailoRT and OpenCV. It's using Hailo Model Zoo module for postprocessing.

**Last HailoRT version checked - 4.12.1**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.


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
