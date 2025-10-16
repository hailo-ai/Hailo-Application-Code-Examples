Instance Segmentation
=====================

This example performs instance segmentation using a **Hailo8** or **Hailo10** device.  
It processes input images, videos, or a camera stream, performs inference using the input HEF file, and overlays the segmentation masks, bounding boxes, class labels, and confidence scores on the resized output image.  
Optionally, object tracking across frames can be enabled for video and camera streams.

![output example](instance_segmentation_example.gif)

Requirements
------------
- hailo_platform==4.22.0
- loguru
- opencv-python
- scipy
- lap
- cython_bbox


Supported Models
----------------
yolov5n-seg, yolov5s-seg, yolov5m-seg, yolov5l-seg, yolov8n-seg, yolov8s-seg, yolov8m-seg, fast_sam_s.

Usage
-----
To avoid compatibility issues, it's recommended to have a separate venv from the DFC.

0. Install PCIe driver and PyHailoRT
    - Download and install the PCIe driver and PyHailoRT from the Hailo website
    - To install the PyHailoRT whl:
        ```shell script
        pip install hailort-X.X.X-cpXX-cpXX-linux_x86_64.whl
        ```

1. Clone the repository:
    ```shell script
    git clone <https://github.com/hailo-ai/Hailo-Application-Code-Examples.git>

    cd Hailo-Application-Code-Examples/runtime/hailo-8/python/object_detection
    ```

2. Install dependencies:
    ```shell script
    pip install -r requirements.txt
    ```

3. Download example files:

   The script supports both Hailo-8 and Hailo-10 files.  
   Use the `--arch` flag to specify your target hardware:
   ```shell
   ./download_resources.sh --arch 8     # For Hailo-8
   ./download_resources.sh --arch 10    # For Hailo-10
    ```

4. Run the script:
    ```shell script
    ./instance_segmentation.py -n <model_path> -i <input_image_path> -a <arch> -b <batch_size>
    ```

You can choose between:
- **Regular instance segmentation**
- **instance segmentation with tracking** (by adding `--track`)

The output results will be saved under a folder named `output`, or in the directory specified by `--output-dir`.



Arguments
---------

- `-n, --net`: Path to the pre-trained model file (HEF).
- `-i, --input`: Path to the input (image, folder, video file, or `camera`).
- `-b, --batch_size`: [optional] Number of images in one batch. Defaults to 1.
- `-l, --labels`: [optional] Path to a text file containing class labels. If not provided, default COCO labels are used.
- `-s, --save_stream_output`: [optional] Save the output of the inference from a stream.
- `-o, --output-dir`: [optional] Directory where output images/videos will be saved.
- `-r, --resolution`: [Camera input only] Choose output resolution: `sd` (640x480), `hd` (1280x720), or `fhd` (1920x1080). If not specified, native camera resolution is used.
- `--track`: [optional] Enable object tracking across frames using BYTETracker.
- `--show-fps`: [optional] Display FPS performance metrics for video/camera input.

For more information:
```shell script
./instance_segmentation.py -h
```

Example 
-------
**Regular object detection on a camera stream**
```shell script
./instance_segmentation.py -n yolov5m_seg_with_nms.hef -i camera -a v5
```

**Object detection with tracking on a camera stream**
```shell script
./instance_segmentation.py -n yolov5m_seg_with_nms.hef -i camera -a v5 --track
```

**Inference on an image**
```shell script
./instance_segmentation.py -n yolov5m_seg_with_nms.hef -i zidane.jpg -a v5
```

**Inference on a folder of images**
```shell script
./instance_segmentation.py -n yolov5m_seg_with_nms.hef -i input_folder -a v5
```

🔧 Visualization and Tracking Configuration
-------------------------------------------
The application supports flexible configuration for how detections and tracking results are visualized. These settings can be modified in the configuration file to adjust the appearance of detection outputs and the behavior of the object tracker.

#### Example Configuration:
```json
"visualization_params": {
    "score_thres": 0.42,
    "mask_thresh": 0.2,
    "mask_alpha": 0.7,
    "max_boxes_to_draw": 30,
    "tracker": {
        "track_thresh": 0.01,
        "track_buffer": 30,
        "match_thresh": 0.9,
        "aspect_ratio_thresh": 2.0,
        "min_box_area": 500,
        "mot20": false
    }
}
```

#### Parameter Descriptions:

**Visualization Parameters:**

- `score_thres`: Minimum confidence score required to display a detected object.
- `mask_thresh`: Threshold for displaying segmentation masks (e.g., only show masks with values above this).
- `mask_alpha`: Transparency level of the segmentation mask overlay (0 = fully transparent, 1 = fully opaque).
- `max_boxes_to_draw`: Maximum number of detected objects to display per frame.

**Tracker Parameters:**

- `track_thresh`: Minimum score for a detection to be considered for tracking.
- `track_buffer`: Number of frames to retain lost tracks before deleting them.
- `match_thresh`: IoU threshold used to associate detections with existing tracks.
- `aspect_ratio_thresh`: Maximum allowed aspect ratio of detected objects (used to filter invalid boxes).
- `min_box_area`: Minimum area (in pixels) of a detection to be considered valid for tracking.
- `mot20`: Whether to use MOT20-style tracking behavior (set to `false` for standard tracking).

📊 Performance Notes
--------------------
This example supports two types of models:

- Models with built-in HailoRT postprocessing (including NMS e.g Yolov5m-seg):  
These models include optimized NMS and postprocessing inside the HEF, allowing full offload to the Hailo device.

      Camera input: ~30 FPS
      Video input: ~42 FPS


- Models that require host-side postprocessing (e.g yolov8s_seg):
   These models rely on the host CPU for NMS and mask refinement, which significantly affects real-time performance.

      Camera input: ~5 FPS
      Video input: ~3 FPS

Additional Notes
----------------

- The example was only tested with `HailoRT v4.22.0`
- Images are only supported in the following formats: .jpg, .jpeg, .png or .bmp
- Number of input images should be divisible by `batch_size`
- Using the yolov-seg model for inference, this example performs instance segmentation, draw detection boxes and add a label to each class. When using the FastSAM model, it only performs the instance segmenation.
- As the example, as mentioned above, made to work with COCO trained yolo-seg models, when using a customly trained yolo-seg model, please notice that some values may need to be changed in the relevant functions AND that the classes under CLASS_NAMES_COCO in hailo_model_zoo/core/datasets/datasets_info.py file in the Hailo Model Zoo are to be changed according to the relevant classes of the custom model.
- For any issues, open a post on the [Hailo Community](https://community.hailo.ai)


Disclaimer
----------
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.
