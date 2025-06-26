Super resolution 
================

This example performs super resolution using a Hailo8 device.
It receives an input image and enhances the image quality and details.

![output example](./output_example.png)

Requirements
------------

- hailo_platform==4.21.0
- loguru
- Pillow
- opencv-python

Supported Models
----------------

- real_esrgan_x2
 
Usage
-----

0. Install PyHailoRT
    - Download the HailoRT whl from the Hailo website - make sure to select the correct Python version. 
    - Install whl:
        ```shell script
        pip install hailort-X.X.X-cpXX-cpXX-linux_x86_64.whl
        ```

1. Clone the repository:
    ```shell script
    git clone <https://github.com/hailo-ai/Hailo-Application-Code-Examples.git>
        
    cd Hailo-Application-Code-Examples/runtime/hailo-8/python/super_resolution
    ```

2. Install dependencies:
    ```shell script
    pip install -r requirements.txt
    ```

3. Download example files:
    ```shell script
    ./download_resources.sh
    ```

4. Run the script:
    ```shell script
    ./super_resolution -n <model_path> -i <input_image_path> -o <output_path> 
    ```

Arguments
---------

- ``-n, --net``: Path to the pre-trained model file (HEF).
- ``-i, --input``: Path to the input image on which super resolution will be performed.
- ``-o, --output``: Path to save the output and comparison.

For more information:
```shell script
./super_resolution.py -h
```
Example 
-------
**Command**
```shell script
./super_resolution.py -n ./real_esrgan_x2.hef -i input_image.png
```

Additional Notes
----------------

- The example was only tested with ``HailoRT v4.21.0``
- The script assumes that the image is in one of the following formats: .jpg, .jpeg, .png or .bmp 

Disclaimer
----------
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.

This example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.
