**Last HailoRT version checked - 4.16.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.


# Hailo-8 Python Inference with OpenCV

This example demonstrates running classification inference using HailoRT's python API and OpenCV to process the input images. 


## Setup

### Hailo-8

Confirm the Hailo-8 PCIe Module has been detected

```bash
lspci
```
<pre id=term>04:00.0 Co-processor: Hailo Technologies Ltd. Hailo-8 AI Processor (rev 01)</pre>


### Environment
A virtual environment must be activated. Either enter the suite docker or activate the DFC's venv. If using the DFC, make sure that the correct pyhailort is installed in that environment.

### Prerequesities:  
numpy  
zenlog  
Pillow
hailo_platform >= 4.14.0 (installed from the HailoRT .whl from the Hailo website or already installed in the Suite docker release of 10.2023)  

Install the hailort whl. (if you are not using the Suite docker), and then the requirements:
`pip install -r requiremets.txt`

## Running
`./run_inference_with_image_opencv.py resnet_v1_50.hef zidane.jpg [--labels PATH_TO_LABELS_FILE]`
