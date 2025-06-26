![github_code](banner.jpeg)
 

## 🚀 [Runtime](https://github.com/hailo-ai/Hailo-Application-Code-Examples/tree/main/runtime)
Application examples for Hailo inference with different languages and operating systems
### Hailo-8 

<div style="margin-left: 20px;">
<details>
<summary>C++</summary>

APP | Description |
|:---|:---|
| `classifier` | Classification with models trained on ImageNet
| `depth_estimation` | Depth estimation with scdepthv3 and stereonet
| `instance_segmentation` | Instance segmentation with yolov5_seg, yolov8_seg
| `object_detection` | Object detection - generic, asynchronous
| `onnxruntime` | Inference with a Hailo device and postprocessing with ONNXRuntime
| `pose_estimation` | Pose estimation with yolov8
| `semantic_segmentation` | Semantic segmentation with Resnet18_fcn trained on cityscape
| `zero_shot_classification` | Zero-Shot Classification with clip_vit_l14

</details>

<details>
<summary>Python</summary>

APP | Description |
|:---|:---|
| `instance_segmentation` | Instance segmentation with yolov5_seg, yolov8_seg
| `lane_detection` | Lane detection with UFLDv2
| `object_detection` | Object detection with yolo, ssd, centernet
| `pose_estimation` | Pose estimation with yolov8
| `super_resolution` | Super resolution with espcnx4, srgan

</details>

<details>
<summary>GStreamer</summary>

APP | Description |
|:---|:---|
| `advanced_cpp_app` | Complex GStreamer pipeline wrapped by C++
| `cropper_aggregator` | Gstreamer pipeline with hailocropper and hailoaggregator
| `detection_python` | Python implementation of TAPPAS detection pipeline using Yolov5m
| `hailo"_clip` | CLIP inference on a video in real-time
| `cpp_cascaded_networks_dynamic_osd` | Cascade networks pipeline wrapped by C++
| `multistream_app` | Inference on multiple streams on the same pipeline, added C++ usability
| `multistream_multi_networks` | Object detection + semantic segmentation
| `multistream_stream_id` | Multistream with stream ID
| `simple_cpp_app` | Simple app that shows how to use Gstreamer with C++ on top
| `tda4vm/pose_estimation` | Single-stream pose estimation pipelin` on top of GStreamer and TDA4VM DSP
| `tonsofstreams` | Many streams with 4 Hailo devices


</details>

<details>
<summary>Windows</summary>

APP | Description |
|:---|:---|
| `yolov5` | Object detection with yolov5 using a C++ script compiled for Windows
| `yolov8` | Object detection with yolov8 using a C++ script compiled for Windows

</details>
</details>
</div>

<br>

### Hailo-15

APP | Description |
|:---|:---|
| `dsp example` | Demonstrate how to use Hailo DSP 
| `zero shot classification` | Zero-Shot Classification with clip_resnet50

</details>

<br>

## 🏗️ [Compilation](https://github.com/hailo-ai/Hailo-Application-Code-Examples/tree/main/compilation)

**Basic optimization diagnostic tool:** help diagnosing common optimization issues and mistakes
</br>**Pointpillars:** Hailo device offload of the heavy 2D-convolutional part of a 3D-object-detection network operating on point-clouds
</br>**16-bit Optimization:** Guide on how to perform 16-bit optimization

<br>

## ⚠️ Disclaimer

The code examples are provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running the examples, please open a ticket in the "Issues" tab.
