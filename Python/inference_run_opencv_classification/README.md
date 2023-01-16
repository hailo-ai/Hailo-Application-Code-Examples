# Hailo-8 Python Inference with OpenCV

This example demonstrates running inference using HailoRT's python API and OpenCV to process the input images. It receives as input the hef of a classification model trained with ImageNet and the path to the input image set.



## Setup

### Hailo-8

Confirm the Hailo-8 PCIe Module has been detected

```bash
lspci
```
<pre id=term>04:00.0 Co-processor: Hailo Technologies Ltd. Hailo-8 AI Processor (rev 01)</pre>


### Environment
A virtual environment must be activated. Either enter the suite docker or activate the DFC's venv. If using the DFC, make sure that the correct pyhailort is installed in that environment.



### Required packages
#### Zenlog
```bash
pip install zenlog
```
#### OpenCV
```bash
pip install opencv-python
```

## Running
```bash
python run_inference_with_image_opencv.py --hef mobilenet_v3_large_minimalistic.hef --input-images ./images/
```
