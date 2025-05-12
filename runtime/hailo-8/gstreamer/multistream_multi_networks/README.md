Multistream object detection + semantic segmentation for autonomous parking
================

This example performs a Gstreamer object detection + semantic segmentation using a Hailo8 device.
It receives 4 input videos, annotates it with detected objects and also performs semantic segmentation for an autonomous parking scenario.

Requirements
------------

Run from TAPPAS docker. 

Supported Models
----------------

This example expects the two models - yolov8s for object detection and fcn8_resnet_v1_18 for semantic segmentation. 

Usage
-----

0. Download the TAPPAS dokcer from the Hailo website
1. Unzip the downloaded file and run the TAPPAS docker - `./run_tappas_docker.sh --tappas-image /path/to/hailo_docker_tappas_VERSION.tar`
2. In the docker, create a new folder under the path /local/workspace/apps/h8/gstreamer/general
3. Copy the demo.sh & get_resources.sh to the new folder in the docker
4. Get the resource by running ./get_resources.sh
5. Create a resources folder for the HEFs and videos: `mkdir -p multistream_app/resources && cd multistream_app/resources && mkdir hefs videos` and move the corresponding files under the relevant folder.
6. Install VA-API for the target machine in the Docker:`./install_accelerator.sh` (can be found under ``/local/workspace/apps/h8/gstreamer/x86_hw_accelerated/`` folder. Might need to run `sudo chmod 777 install_accelerator.sh` before running it).
7. Set the relevant VA-API environment variables -<br/> 
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/x86_64-linux-gnu/`<br/>
`export LIBVA_DRIVER_NAME=iHD`<br/>
`export GST_VAAPI_ALL_DRIVERS=1`<br/>
8. Run the demo - `./demo.sh`


Additional Notes
----------------

- The example was only tested with ``TAPPAS v3.28.0``
- The example expects the two specific HEFs that are provided
- The example outputs the results to the screen 

Disclaimer
----------
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.

This example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.
