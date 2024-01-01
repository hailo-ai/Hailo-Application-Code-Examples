**Last HailoRT version checked - 4.16.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.



## espcn inference example
This example shows SR (Super-Resolution) inference example.  
It uses the model espcn_x4_540_960.hef
It takes image in 540x960 resolution and enlarged it to 4k resolution.

## Prerequesities: 
Pillow
loguru
opencv-python
hailo_platform (installed from the HailoRT .whl of version >= 4.13.0)  

## Usage
```
./espcnx4_infer.py espcn_x4_540x960.hef --image images/ -o output_dir
```   

