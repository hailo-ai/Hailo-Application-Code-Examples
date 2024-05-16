# Hailo-15 example for multi-network pipeline with dynamic OSD  
## Preivew 
![gif](https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/images/eran_pose.gif)

## Features
#### This example application demonstrates the following features
###### - Running a Gstreamer code wrapped by CPP
###### - Multiple rescale from frontend (Sensor's 4K to FHD, HD & VGA)
###### - Yolov5s object detection with postprocessing for person class only 
###### - mspn regnetx pose estimation network for the cropped human detections
###### - Yolov8s object detection with postprocessing for all classes but a person
###### - Hailo Tracker
###### - Dynamic Update of OSD (Angular change)  

## Pipeline
![github_code](https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/images/h15_cascaded_networks_dyn_osd.png)
<br/>
## Prerequisites: <br />
In order to compile this application you'll need the H-15 cross-development toolchain.
The toolchain can be installed by running the script in the `sdk` folder which is a part of the Vision Processor Software Package.<br/>
In order to run this application you'll need a Hailo-15 platform connected with ethernet to your development machine.
For more information please check the EVB/SBC Quick Start Guide.
<br/>

## Running this example (Tested on Release 1.3.0)
* On your development machine
  * Get the resources <code> ./get_resources.sh </code>
  * The 3 .hef files should have been downloaded and placed under the `resources/` folder
  * Compile the example for Hailo-15 <code> ./build.sh /opt/poky/4.0.2/ </code>
* On your Hailo-15 platform
  * Create a folder for this example <code> mkdir ~/apps/cascaded_networks_dynamic_osd</code>
  * Copy the resources folder from the development machine to the folder you created 
  * Copy the application binary (from the development machine) `build/aarch64/cascaded_networks_dynamic_osd` to the folder you created
  * Copy the postprocess binary (from the development machine) `build/aarch64/libyolo_hailortpp.so` to the resources folder on the Hailo-15 platform
  * Run the binary <code> ~/apps/cascaded_networks_dynamic_osd/cascaded_networks_dynamic_osd </code>
* On your development machine
    * Receive and display the incoming stream <code> ./udp_stream_display.sh </code>
