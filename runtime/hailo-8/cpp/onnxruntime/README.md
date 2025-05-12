**ONNXRutime(ORT) inference pipeline**
======================================

This is a Hailo-ONNXRutime(ORT) inference pipeline code example.
Use ONNX Runtime to efficiently run inference as a post-processing step if there are operations that we are not supporting them. <br/>
The example consists of two different consecutive parts:

    1. Hailo inference
    2. ORT inference (postprocessing)

Requirements
------------
HailoRT == 4.18.0

**_NOTE:_** Currently supports only devices connected on a PCIe link.

Usage
-----
1. Download example files:
	```shell script
    ./get_hef_and_onnx.sh 
    ```
2. Compile the project on the development machine  
	```shell script
    ./build.sh
    ```
3. Run the example:

	```shell script
    ./build/x86_64/hailo_ort_example -hef=yolov5m_wo_spp.hef -onnx=yolov5m_wo_spp_postprocess_v1.onnx -image=<path to your image> -num=1
    ```

Arguments
---------

- ``-input``: Path to the input image\video\camera on which object detection will be performed.
- ``-hef``: Path to HEF file to run inference on.
- ``-onnx``: Path to ONNX file to run inference on.
- ``-num (optional)``: Number of frames to run inference on.

**_NOTE:_** When running ./get_hef_and_onnx.sh, the onnxruntime C++ tar package v1.13.0 will be downloaded and extracted at path /opt. In case you don't have premissions to perform changes in this folder or you wish to have it extracted somewhere else, please change the get_hef_and_onnx.sh bash script accordingly AND update the onnxruntime folder path inside the CMakeLists.txt file. 

Disclaimer
----------
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.

This example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.

**Last HailoRT version checked - 4.18.0**