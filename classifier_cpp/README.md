
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

