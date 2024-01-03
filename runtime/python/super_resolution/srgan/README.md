## SRGAN inference example
This example shows SR (Super-Resolution) inference example.  
It uses the model srgan.hef.  

## Prerequesities: 
Pillow
loguru
opencv-python
hailo_platform (installed from the HailoRT .whl of version >= 4.13.0) 

## Usage
```
./srgan_infer.py --hef srgan.hef --images test_images/ --show
```   
![srgan_compare](https://user-images.githubusercontent.com/88292552/213993733-0e45adeb-3e64-4531-97b7-a0b029036ec4.PNG)
