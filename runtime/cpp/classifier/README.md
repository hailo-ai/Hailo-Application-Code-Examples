**Last HailoRT version checked - 4.18.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.

 C++ Classifier Example
--------------------------------------------------

This example uses the C++ API of HailoRT to implement a classifier that was trained on the 
ImageNet dataset (1000 classes). The inputs to the code a compiled network (HEF) file and
a directory containing the image files to classify.

1. Dependencies:
    - OpenCV, and g++-9: 
    ``` bash
    sudo apt-get install -y libopencv-dev gcc-9 g++-9
    ```
2. Downloads the hef and sample image files:
   `./get_hef.sh`

3. Build the project build.sh

4. Run the executable:
    ``` bash
	./build/x86_64/classifier -hef=resnet_v1_50.hef -path=./
    ```


 Classifier customization
-------------------------------------------------
This example assumes that the classifer was trained on ImageNet, with an input of 224x224. If 
the input size is different, please change the below to match real input resolution:

``` cpp
constexpr int WIDTH  = 224;
constexpr int HEIGHT = 224;
```

The example assumes that the Softmax layer is part of the graph that runs on the Hailo device.
If this is not the case, please change the call to `classification_post_process` to reflect 
that.

