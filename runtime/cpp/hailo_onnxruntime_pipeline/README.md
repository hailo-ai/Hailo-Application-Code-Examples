**Last HailoRT version checked - 4.15.0**

This is a Hailo-ONNXRutime(ORT) inference pipeline code example.

The example consists of two different consecutive parts:

    1. Hailo inference
    2. ORT inference (postprocessing)

**_NOTE:_** Currently supports only devices connected on a PCIe link.

Prequisites:
HailoRT >= 4.13.0


To get a HEF file and its postprocess ONNX, run ./get_hef_and_onnx.sh 

**_NOTE:_** When running ./get_hef_and_onnx.sh, the onnxruntime C++ tar package v1.13.0 will be downloaded and extracted at path /opt. In case you don't have premissions to perform changes in this folder or you wish to have it extracted somewhere else, please change the get_hef_and_onnx.sh bash script accordingly AND update the onnxruntime folder path inside the CMakeLists.txt file. 

The command line arguments are as follows:

- -hef= The input path of the HEF file
- -onnx= - The path to the ONNX file.
- -num= (optional) - The number of frames to run inference on. If no number os supplied the default is 100. 
- -image= (optional) - The image file path for the image to perform the inference on. The example currently supports only one input image.

To compile the example run ./build.sh

To run the compiled example:
./build/x86_64/hailo_ort_example -hef=HEF_PATH.hef -onnx=ONNX_PATH.onnx [-num=NUM_OF_FRAMES -image=IMAGE_PATH]

**_NOTE:_** If there is a space after the "=" sign (for example: -onnx= ResNet18.onnx) the argument will not catch and you will get an error \ wrong results.

**_NOTE:_** The default run is without any input images. If you want to use an actual image, comment in all the line under "WITH IMAGES" and comment out the respective lines for the default run.

**_NOTE:_** This example works either with random data or a single image. If you wish for it to work with multiple images or a video, you will need to implement it.

**_NOTE:_** This example only include the Hailo inference and ONNX postprocess part of a specific HEF file and a specific ONNX file. If you wish to have additional operations and functions, for example drawing the model's results on an image, you need to implement it according to the model that is run.
