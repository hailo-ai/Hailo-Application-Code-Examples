# Raw Streams Async Example (yolov5)  

This example demonstrates the use of HailoRT async API on raw streams (not VStreams).

To get hef and video:  
``
get_hef_and_video.sh
``

To build for x86_64:  
Use CMakeLists_x86.txt (change it's name to CMakeLists.txt)  
``
./build.sh
``
To build for hailo-15:  
Use CMakeLists_h15.txt (change it's name to CMakeLists.txt)   
Add in multi_async.cpp: ```#define _EMBEDDED_ // add to enable embedded mode (jpg instead of mp4, for h15 / imx) ```   
and then run:  
``
#enable cross-compile environment
. /opt/poky/4.0.2/environment-setup-armv8a-poky-linux
./build.sh
``

To run:  
```
./build/async_infer
  
# open processed video  
vlc processed_video.mp4  
```

## Notes:  
* If you do not want to save the processed video as mp4, 
comment out the line ```#define SAVE_TO_FILE // comment out to disable saving to file ``` in ```multi_async.cpp```.  
* If you compile on embedded system which doesn't have a decoder (such as hilo-15), please add ```#define _EMBEDDED_```  in ```multi_async.cpp```.  

## Overview  
This code represents an application for performing inference using HailoRT async API on raw-streams (not VStreams). The application is designed to process video data from a video file, run it through a YOLOv5 model, and post-process the results to detect objects in the video frames.

## Code Structure  
The code consists of several classes and threads that work together to perform the inference and object detection. Here is an overview of the code structure:

**AbstractCapture**  
This absract class is responsible for capturing video frames from a video file (VideoCapture) or image file (ImageCapture).  
It provides methods to get the next frame, set the height and width of frames, and retrieve frame dimensions.  
In the case of VideoCapture- it will process all frames in the video file.  
In the case of ImageCapture- it will process ```default_max_num_frames_to_process``` times the same image (for fps calculation).
**App**  
The main application class that orchestrates the entire process.  
It includes members for managing the video input, the Hailo vdevice, and other configurations.  
The class also handles multiple threads for input, output, and post-processing.  

**Threads**  
The application uses multiple threads to efficiently process video frames and perform inference:  

**Input Thread**:  
Captures video frames and sends them for inference.  
Monitors the status of the input stream.  
**Output Threads** (Multiple):  
Each output thread reads inference results from a specific output stream.  
Monitors the status of the output streams.  
**Post-Processing Thread**:  
Receives inference results from the output threads.  
Performs post-processing on the results to detect objects.  
Draws bounding boxes around detected objects on the frames.  

**Output**    
Detected objects are highlighted with bounding boxes on the frames, and saved to processed_video.mp4.  
Also, if set (const bool print = true in main), The application prints information about the detection of objects in the video frames, including the class ID, bounding box coordinates, and confidence score.

## Dependencies
The code uses the following external libraries and components:  
**OpenCV** (for capturing and processing video frames)  
**HailoRT** 4.14.0 (for running inference with a Hailo deep learning accelerator)  
Please ensure that these dependencies are properly installed and configured to run the application successfully.  

**Performance on x86**  
FPS > 200, CPU util < 165%  
