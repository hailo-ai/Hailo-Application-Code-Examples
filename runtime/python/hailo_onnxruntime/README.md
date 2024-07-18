**Last HailoRT version checked - 4.16.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.


This is a general Hailo + ONNXRuntime inference example.  

The example takes one or more images, performs inference using the input HEF file and then finishing the postprocessing using an ONNX file.  
The example works with .jpg, .jpeg, .png and .bmp image files.  

## Prerequesities:  
numpy  
zenlog  
Pillow
onnxruntime
onnx
onnxruntime-openvino
 
hailo_platform >= 4.14.0 (installed from the HailoRT .whl from the Hailo website or already installed in the Suite docker release of 10.2023)  

Install the hailort whl. (if you are not using the Suite docker), and then the requirements:
`pip install -r requiremets.txt`


## Running the example:  
```./hailo_onnxruntime_inference.py HEF.hef ONNX.onnx [--input-images PATH_TO_IMAGES_FOLDER_OR_IMAGE] [--accelerate]```

You can download a sample image and a HEF with the `get_hef_and_onnx.sh` script, and then execute the inference.
for example:  
```CUDA_VISIBLE_DEVICES=9 ./hailo_onnxruntime_inference.py yolov5m_wo_spp.hef yolov5m_wo_spp_postprocess.onnx```

Notice: By adding the flag ``--accelerate``, you can use OpenVINO inference acceleration. Note that the OpenVINO acceleration will be useful only for longer ONNX inference durations, but would be less efficient for short inference durations. 


For more information, run ```./hailo_onnxruntime_inference.py --help```   
