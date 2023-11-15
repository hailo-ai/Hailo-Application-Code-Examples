# Mobilenet-SSD-v2 With NMS On Host Cpp Code Example  

**Last HailoRT version checked - 4.15.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.


SSD detection example with HailoRT C++, NMS on host.  

## How to run  
1. To compile the example run `./build.sh`     
2. get hef and video: `./get_hefs_and_video.sh`   
For better performance, resize offline by running: `ffmpeg -i full_mov_slow.mp4 -vf scale=300:300 full_mov_slow_scaled.mp4`  
3. To run the compiled example:  
`./build/x86_64/vstream_ssd_example_cpp -hef=./ssd_mobilenet_v2_wo_nms.hef -video=./full_mov_slow_scaled.mp4`  

### Notes  
1. You can also save the processed video by commenting in a few lines in the "post_processing_all" function.  
2. There should be no spaces between "=" given in the command line arguments and the file name itself.  

## Prerequirements  
1. OpenCV 4.2.X  
2. CMake >= 3.20  
3. HailoRT >= 4.12.0  
NOTE: Currently supports only devices connected on a PCIe link. 

## What this examploe does?  

The example does the following:  

1. Creates a device (pcie)  
2. Reads the network configuration from ssd HEF file  
3. Prepares the application for inference  
4. Runs inference and postprocess on a given video file  
5. Draws the detection boxes on the frame  
6. Prints the object detected + confidence to the screen  
5. Prints statistics  
