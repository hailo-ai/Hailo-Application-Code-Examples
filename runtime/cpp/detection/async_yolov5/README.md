**Last HailoRT version checked - 4.15.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.

# Raw Streams Async Example (yolov5)  

This example demonstrates the use of HailoRT async API on raw streams (not VStreams).

## To get hef and video:  
``
get_hef_and_video.sh
``
  
## To build for x86_64:  
* Use CMakeLists_x86.txt (change it's name to CMakeLists.txt)  
* And then run:  
``
./build.sh
``
## To build for hailo-15:  
* Use CMakeLists_h15.txt (change it's name to CMakeLists.txt)   
* Add in multi_async.cpp: ```// #define _EMBEDDED_ // add to enable embedded mode (h15) ```   
* And then run:  
```
# enable cross-compile environment  
. /opt/poky/4.0.2/environment-setup-armv8a-poky-linux  
./build.sh  
```

## To run:  
```
./build/async_infer
  
# if x86_64, open processed video  
vlc processed_video.mp4  
```
## Performance:  
|                         | x86_64 (laptop) | h15            |
| ------------------------|:---------------:|:--------------:|
| cpu util on peak        | 165 (/1600)     | 63.8 (/400)    |
| fps                     | 217             | 68             |
| hailortcli run hw_only  | 217.78          | 70             |

## Notes:  
* If you do not want to save the processed video as mp4, 
comment out the line ```#define SAVE_TO_FILE // comment out to disable saving to file ``` in ```multi_async.cpp```.  
* If you compile on embedded system which doesn't have a decoder (such as hailo-15), please add ```#define _EMBEDDED_```  in ```multi_async.cpp``` and use the appropriate ``CMakeLists.txt``.  
* yolov5m_wo_spp_60p_async_h15.hef includes ```tf_rgb_to_hailo_rgb``` and ```hailo_rgb_to_tf_rgb``` format conversions.  

## Overview  
This code represents an application for performing inference using HailoRT async API on raw-streams (not VStreams). The application is designed to process video data from a video file or jpg file, run it through a YOLOv5m model, and post-process the results to detect objects in the video frames.

## Code Structure  
The code consists of several classes and threads that work together to perform the inference and object detection. Here is an overview of the code structure:

**AbstractCapture**  
This absract class is responsible for capturing video frames from a video file (VideoCapture) or image file (ImageCapture).  
It provides methods to get the next frame, set the height and width of frames, and retrieve frame dimensions.  
In the case of VideoCapture- it will process all frames in the video file.  
In the case of ImageCapture- it will process ```default_max_num_frames_to_process``` times the same image (for fps calculation).  

**App**  
The main application class that orchestrates the entire process.  
It includes members for managing the video or image input, the Hailo vdevice, and other configurations.  
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
On x66_64, detected objects are highlighted with bounding boxes on the frames, and saved to processed_video.mp4.  
Also, if set (const bool print = true in main), The application prints information about the detection of objects in the video frames, including the class ID, bounding box coordinates, and confidence score.

## Dependencies
The code uses the following external libraries and components:  
**OpenCV** (for capturing and processing video frames)  
**HailoRT** 4.14.0 (for running inference with a Hailo deep learning accelerator)  
Please ensure that these dependencies are properly installed and configured to run the application successfully.  

**Performance on x86**  
FPS > 217, CPU util < 10.3%  
