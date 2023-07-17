
C++ Segmentation Inference Example
--------------------------------------------------

This example uses the C++ API of HailoRT to implement a segmentation inference example, suitable for Resnet18_fcn, and for cityscape images. 
The inputs to the code a compiled network (HEF) file and
a video file (.mp4 or .avi) to be processed.
The output is a processed segmented video file - 'processed_video.mp4'

This example was tested on this setup:
- x86 machine
- M-key Hailo-8
- HailoRT 4.12.0 


1. Dependencies:
    - OpenCV, and g++-9:
    ``` bash
    sudo apt-get install -y libopencv-dev gcc-9 g++-9
    ```
2. Build the project build.sh
3. Run the executable:
    ``` bash
    ./build/segmentation_example_cpp -hef=HEF_PATH -path=VIDEO_PATH
    ```
    This example contains the hef file fcn16_resnet_v1_18.hef and the video full_mov_slow.mp4, so for a quick demo you can run:
    ``` bash
    ./build/segmentation_example_cpp -hef=fcn16_resnet_v1_18.hef -path=full_mov_slow.mp4
    ```

Segmentation example customization
--------------------------------------------------
This example assumes that the input and outputs of the network is in UINT8 format. 
If the input\outputs format is different, please change the below (marked with ** **) to match real input\outputs format:

in 'main()':
``` cpp
auto input_vstream_params = network_group.value()->make_input_vstream_params(false, ** HAILO_FORMAT_TYPE_UINT8 **,HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);

auto output_vstream_params = network_group.value()->make_output_vstream_params(false, ** HAILO_FORMAT_TYPE_UINT8 **, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);

auto status  = infer<uint8_t, uint8_t>(vstreams.first, vstreams.second, video_path);
```

``` cpp
in 'read_all()':
seg_image.convertTo(seg_image, ** CV_8U **, 1.6, 10);
```
in 'write_all()':
int factor = std::is_same<T, ** uint8_t **>::value ? 1 : 4;  // In case we use float32_t, we have 4 bytes per component
