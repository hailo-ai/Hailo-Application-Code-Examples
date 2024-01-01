**Last HailoRT version checked - 4.16.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.


This is a StereoNet Hailo inference example.

The example takes one or more pair of images and performs inference using the StereoNet input HEF file.
The example works with .jpg, .png and .bmp image files.

## Prerequesities: 
tensorflow
zenlog  
Pillow  
psutil    
hailo_platform (installed from the HailoRT .whl of version >= 4.14.0)  

## Install Prerequesities (after installinf hailo_platform from the HailoRT .whl):
```pip install -r requirements.txt```

## Running the example:
```./stereo_inference_run.py HEF_PATH --right RIGHT_IMAGES_PATH --left KEFT_IMAGES_PATH [--output-path path/to/output/images/folder]```

You can get example HEF files to test the example with by running ```./get_hef.sh``` and the run:
```./stereo_inference_run.py stereonet.hef --right right.jpg --left left.jpg```

NOTE: Since this example is based on using the stereonet HEF file from the Hailo Model Zoo, the left image goes to "input_layer1" and the right image goes to "input_layer2". This might not be the case for other StereoNet models, so in case you try a different HEF and not the default one, please perform the relevant changes in the code.

For more information, run ```./stereo_inference_run.py -h```
