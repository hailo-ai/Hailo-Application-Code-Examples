![github_code](banner.jpeg)

Welcome to the Hailo Application Code Examples repository! <br>
Here you'll find a collection of application examples for running inference on Hailo-8, Hailo-8L, Hailo-10, and Hailo-15 devices, using various programming languages and operating systems.

---

## 🚀 [Runtime](https://github.com/hailo-ai/Hailo-Application-Code-Examples/tree/main/runtime)

Application examples for real-time inference on Hailo accelerator devices.

<details>
<summary><strong>C++ Examples</strong></summary>

| APP                        | Description                                                      |
|:---------------------------|:-----------------------------------------------------------------|
| `classification`               | Image classification with models trained on ImageNet         |
| `depth_estimation`         | Depth estimation using scdepthv3 and stereonet                   |
| `instance_segmentation`    | Instance segmentation with yolov5_seg and yolov8_seg             |
| `object_detection`         | Generic and asynchronous object detection                        |
| `onnxruntime`              | Inference with Hailo device and postprocessing via ONNXRuntime   |
| `pose_estimation`          | Pose estimation with yolov8                                      |
| `semantic_segmentation`    | Semantic segmentation with Resnet18_fcn (Cityscapes dataset)     |
| `zero_shot_classification` | Zero-shot classification with clip_vit_l14                       |
| `oriented_object_detection`| Oriented object detection using YOLO11 OBB                       |

</details>

<details>
<summary><strong>Python Examples</strong></summary>

| APP                        | Description                                                        |
|:---------------------------|:-------------------------------------------------------------------|
| `object_detection`         | Object detection and tracking with YOLO, SSD, and CenterNet        |
| `instance_segmentation`    | Instance segmentation with yolov5_seg and yolov8_seg               |
| `lane_detection`           | Lane detection using UFLDv2                                        |
| `pose_estimation`          | Pose estimation with yolov8                                        |
| `speech_recognition`       | Automatic speech recognition with the Whisper model                |
| `super_resolution`         | Super-resolution with espcnx4 and SRGAN                            |
| `oriented_object_detection`| Oriented object detection using YOLO11 OBB                         |

</details>


## 📷 [VPU](https://github.com/hailo-ai/Hailo-Application-Code-Examples/tree/main/vpu)

Application examples for Hailo VPU:
<details>
<summary><strong>All VPU Examples</strong></summary>

| APP                      | Description                                                      |
|:-------------------------|:-----------------------------------------------------------------|
| `dsp example`            | Demonstrates usage of the Hailo DSP                              |
| `zero shot classification` | Zero-shot classification with clip_resnet50                   |

</details>

## 🏗️ [Compilation](https://github.com/hailo-ai/Hailo-Application-Code-Examples/tree/main/compilation)

<details>
<summary><strong>All Compilation Examples</strong></summary>

| APP                              | Description                                                                 |
|:----------------------------------|:----------------------------------------------------------------------------|
| `basic_optimization_diagnostic`   | Diagnose common optimization issues and mistakes.                           |
| `pointpillars`                    | Offload the heavy 2D-convolutional part of a 3D object detection network (PointPillars) to a Hailo device. |
| `16bit_optimization`              | Guide for performing 16-bit optimization on Hailo devices.                  |

</details>

## ⚠️ Disclaimer

The code examples are provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running the examples, please open a ticket in the "Issues" tab.
