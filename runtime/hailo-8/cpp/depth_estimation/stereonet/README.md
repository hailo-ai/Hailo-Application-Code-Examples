**Last HailoRT version checked - 4.18.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.


This is a hailort C++ API StereoNet stereo depth estimation example.

The example does the following:

1. Creates a device (pcie)
2. Reads the network configuration from a StereoNet HEF file
3. Prepares the application for inference
4. Runs inference on a given image\video file or a camera
5. Prints statistics

**NOTE**: Currently support only devices connected on a PCIe link.

**NOTE**: This example was tested with a stereonet model taken from the Hailo Model Zoo.


Prequisites:

OpenCV 4.2.X

CMake >= 3.20

HailoRT >= 4.10.0


To compile the example run `./build.sh`

To run the compiled example:
`./build/x86_64/stereonet_example_cpp -hef=STREONET_HEF_FILE.hef -right=RIGHT_INPUT -left=LEFT_INPUT`

Example:<br />
`./build/x86_64/stereonet_example_cpp -hef=stereonet.hef -right=right.jpg -left=left.jpg`<br />
Example of directories inputs:<br />
`./build/x86_64/stereonet_example_cpp -hef=stereonet.hef -right=/path/to/right/images/directory/ -left=/path/to/left/images/directory/`


**NOTE**: This example supports single pair of images, pair of videos or multiple images from given folders as inputs. By default, the example would save the inferred images to the current directory. To display the output, comment in\out the relevant lines in the code in the read_all function.

**NOTE**: There should be no spaces between "=" given in the command line arguments and the file name itself.

**NOTE**: In case you use a cemera input, the initial number of frames the example would perform inference on is 300. To change it, update the CAMERA_INPUT_IMAGE_NUM global variable inside the code. 

**NOTE**: Since this example is based on using the stereonet HEF file from the Hailo Model Zoo, the left image goes to "input_layer1" and the right image goes to "input_layer2". This might not be the case for other stereoNet models, so in case you try a different HEF and not the default one, please check the relevant code in run_inference function and change it accordingly if needed. 

