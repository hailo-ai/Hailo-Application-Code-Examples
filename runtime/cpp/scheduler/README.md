**Last HailoRT version checked - 4.16.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.

This is a genereic HailoRT C++ API multi-hef scheduler example for running inference on multiple HEF files using the HailoRT scheduler with vstreams. The example also supports running with the HailoRT Multi-Process Service. <br />
The example does the following:

1. creates a device (pcie)
2. reads the network configuration from a HEF files
3. prepares the application for inference
4. runs inference on all the given networks using the HailoRT scheduler
5. prints statistics

**NOTE**: Currently support only devices connected on a PCIe link.


Prequisites:

OpenCV >= 4.2.X

CMake >= 3.20

HailoRT >= 4.14.0


To compile the example run `./build.sh`


To run the compiled example: <br />
``` ./build/x86_64/switch_network_groups_example -hefs=HEF1.hef,HEF2.hef,... -num=NUM_OF_FRAMES_FOR_HEF1,NUM_OF_FRAMES_FOR_HEF2,... ``` <br />

Example:<br />
```./build/x86_64/scheduler_example -hefs=resnet_v1_50.hef,yolov5m_wo_spp.hef,yolov7.hef -num=150,300,222``` <br />

**NOTE**: The number of HEF files given should match the number of frames counts given. If no number of images is specified, the example would run with a 100 randomly generated images for each HEF.

**NOTE**: When giving the input HEFs\frames count, there should be NO spaces between the arguments.
