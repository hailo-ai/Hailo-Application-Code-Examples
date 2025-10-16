**Last CUDA version checked - 11.3**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.



# Intro

This is a minimalistic proof-of-concept showing how a Hailo device can offload the heavy 2D-convolutional part of a 3D-object-detection network operating on point-clouds. For this example we use the PointPillars (PP) network from OpenPCDet repo, a modular ecosystem with many 3D networks. The pre/post computation is running in PyTorch using native OpenPCDet code. Almost all of it on CPU, sans the 3D-NMS op which unfortunately is compliled for cuda only. 

### Final result
Processing a point cloud to get 3D boxes - similar result when executing part of network on Hailo.
Use the jupyter notebook to go through all steps towards this bottom line.
Original result:

<img src="./rsc/orig_model_front.png" alt="drawing" width="40%"/>

Hailo Assisted:

<img src="./rsc/hailo_assisted_model_front.png" alt="drawing" width="40%"/>

# OpenPCdet - overview
[https://github.com/open-mmlab/OpenPCDet]

<img src="./rsc/openpcdet_1.png" alt="drawing" width="50%"/>
<img src="./rsc/openpcdet_2.png" alt="drawing" width="60%"/>


### We map to Hailo the 2D backbone and detection head:
The purple bracket marks the part of net we can offload to any Hailo device ("Hx" = H8,H15,..). 
In case of PointPillars, that makes the lion's share of TOPS, as 3D part is minimal.The 3D parts of this and other nets can be Hailo-mapped too, but with a significant task-specific effort required to achieve good efficiency.

<img src="./rsc/openpcdet_3.png" alt="drawing" width="50%"/>

# Setup

Run the notebook from a virtualenv prepared like so:

1. Install CUDA and Pytorch. Tested configs:
    1. `pip install torch==1.12.1+cu113` (assuming CUDA 11.3) **OR**
    1. `pip install torch=1.12.1+cu102` (assuming CUDA 10.2)
1. Clone & install OpenPCDet: (tested w. commit a68aaa656 04-Apr-23) 
    ```
     pip install -r requirements
     pip install spconv kornia
     python setup.py develop
   ```
1. Install Mayavi for 3D visualization of point clouds and 3D boxes. Installing and using Mayavi and its dependencies (PyQT5) might be tricky, especially working remotely on a server, so in the notebook code we create visual results as png files without any windows. Still, it might help to prepend the *jupyter-notebook* launch command with instructions to skip checking for gui toolkit (s.a. Qt) and, if no screen ("headless"), also creating and working against a virtual display, like so
```
ETS_TOOLKIT=null xvfb-run --server-args="-screen 0 1024x768x24" jupyter notebook ...
```

1. Download pretrained PointPillar pytorch model from the link below into your *openpcdet-clone-location*:
[https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing]
