**Last HailoRT version checked - 4.16.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.



[UFLDv2] Lane Detection Inference Example
-------------------------------------------

This example performs lane detection inference on an input video, using [UFLDv2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2).

This example was tested with the following setup:
- x86 machine
- M-key Hailo-8
- HailoRT 4.15.0

### How to run the example

1. Install Requirements:
    ``` bash
    pip install -r requirements.txt
    ```
2. Download files
    ``` bash
    ./get_files.sh
    ```
3. Run the example:
    ``` bash
    ./ufld_example.py 
    ```
    This will run the example with the downloaded **ufld_v2.hef** and **input_video.mp4** as inputs. 
    
    The postprocessed video will be saved as **output_video.mp4**. 
    
    For different options:
    ``` bash
    ./ufld_example.py -m MODEL_HEF_PATH -i INPUT_VIDEO_PATH -o OUTPUT_VIDEO_PATH
    ```
