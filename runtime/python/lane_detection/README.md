
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
    pip install -r requiremets.txt
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