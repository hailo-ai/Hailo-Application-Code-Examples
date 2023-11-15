**Last HailoRT version checked - 4.12.1**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.


This is a hailort C++ API yolov8 detection example for ARM imx8 platfroms.

The example does the following:

1. Creates a device (pcie)
2. Reads the network configuration from a yolov8 HEF file
3. Prepares the application for inference
4. Runs inference and postprocess (using xtensor library) on a given image\video file 
5. Draws the detection boxes on the original image\video
6. Prints the object detected + confidence to the screen
5. Prints statistics

**NOTE**: Currently support only devices connected on a PCIe link.

**NOTE**: This example was tested with a yolov8s & a yolov8m model.


Prequisites:

OpenCV 4.5.X

CMake >= 3.16.3 (was tested on 3.16.3, 3.22.0 & 3.24.1)

HailoRT >= 4.12.1

git

Xtensor - no installation or build needed as it's complied from a git repository as an external project.   



To compile the example run `./build.sh`  

To run the compiled example:  

For an image:  
`./build/aarch64/vstream_yolov8_example_cpp -hef=YOLOv8_HEF_FILE.hef -input=IMAGE_FILE.jpg [-num=NUM_OF_IMAGES]`  
For a video:  
`./build/aarch64/vstream_yolov8_example_cpp -hef=YOLOv8_HEF_FILE.hef -input=VIDEO_FILE.mp4 [-num=NUM_OF_IMAGES]`  

**flags description:**  
-hef= - Path to the Yolov8 compiled HEF file.  
-input= - Path to the image or video input file  
-num= (optional) - In case it's used, represents the number of images to run inference on. When using this flag, the example will take only the first frame from the input and run inference on it NUM_OF_IMAGES times.   
If not used, the number of images will be the number of images supplied (1 if image, the total number of frames if video).   

**NOTICE**: This example purpose is to show the abilities of the Hailo-8 chip for Yolov8 models including postprocessing. By default, you should run the example with the "-num" flag to see the performance of the Hailo chip. **This example does not regards the overhead and impact on performance of the decoding or visualiztion**. Please note that this, specifically in ARM machines, can have a great impact if done using software (OpenCV).    


**NOTE**: This example uses xtensor C++ ibrary compiled from the xtl git as an external source.   

**NOTE**: You can also save the processed image\video by commenting in a few lines in the "post_processing_all" function.   

**NOTE**: There should be no spaces between "=" given in the command line arguments and the file name itself.   

**NOTE**: You can play with the values of IOU_THRESHOLD and SCORE_THRESHOLD in the yolov8_postprocess.cpp file for different videos to get more detections.   

**NOTE**: In case you prefer to perform the Sigmoid on host, you can comment in the relevant line to do that. Please notice that you'll need a HEF file that does not have an on-chip sigmoid if you choose to use the example in such a way.   


**IMPORTANT NOTE**: The pre-compiled Yolov8 HEF files in the Hailo Model Zoo are compiled to 8-bit. Hailo supply the option to compile the model with 16-bit output layers for those who desire it. Both scores and data dequantization is done manually in the postprocessing functions. This means that you will not get good detection (or detections at all) with a Yolov8 with 16-bit output layers. If you choose to work with your own HEF that is with a 16-bit output, you need to change the code from uint8_t to uint16_t in the following lines:

double_buffer.hpp - lines 32, 43, 61, 69, 97

yolov8_inference.cpp - line 68

yolov8_postprocess.cpp - lines 78, 82, 139 

hailo_tensors.hpp - lines 19, 29, 46, 85

tensors.hpp - lines 24, 27, 43

