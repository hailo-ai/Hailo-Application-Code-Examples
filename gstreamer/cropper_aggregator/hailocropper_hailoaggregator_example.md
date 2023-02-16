This example was tested on:

- Ubuntu 22.04, 5.15.0-60-generic
- tappas\_3.23.0\_ubuntu22\_docker\_x86\_64
- Dell laptop
    

**Overview**

**HailoCropper** is an element providing cropping functionality. It has 1 sink and 2 sources. HailoCropper receives a frame on its sink pad, then invokes it's `prepare\_crops` method that returns the vector of crop regions of interest `crop\_roi`, for each `crop\_roi` it creates a cropped image (representing it's x, y, width, height in the full frame). The cropped images are then sent to the second src. From the first src we push the original frame that the detections were cropped from.

**HailoAggregator** is an element designed for applications with cascading networks or cropping functionality, meaning doing one task based on a previous task. A complement to the HailoCropper, the two elements work together to form versatile apps. It has 2 sink pads and 1 source: the first sinkpad receives the original frame from an upstream hailocropper, while the other receives cropped buffers from that hailocropper. The HailoAggregator waits for all crops of a given orignal frame to arrive, then sends the original buffer with the combined metadata of all collected crops. HailoAggregator also performs a 'flattening' functionality on the detection metadata when receiving each frame, detections are taken from the cropped frame, copied to the main frame and re-scaled/moved to their corresponding location in the main frame (x,y,width,height).


**Example**

following example will demonstrate the use of the hailocropper and hailoaggregator in a Gstreamer pipeline which overlay a detection bbox over the original frame and send the output to display.

`gst-launch-1.0 filesrc location=/local/workspace/tappas/apps/gstreamer/resources/mp4/street_night.mp4 name=src_0 ! \ decodebin ! videoconvert ! \ hailocropper so-path=/local/workspace/tappas/apps/gstreamer/libs/post_processes//cropping_algorithms/libwhole_buffer.so \ use-letterbox=false function-name=create_crops internal-offset=true name=person_detect_cropper \ hailoaggregator name=person_detect_agg person_detect_cropper.src_0 ! \ queue name=person_detect_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \ person_detect_agg.sink_0 person_detect_cropper.src_1 ! \ queue name=hailo_person_detect_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \ video/x-raw, pixel-aspect-ratio=1/1,format=NV12 ! \ queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! videoconvert n-threads=2 qos=false ! \ queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \ hailonet hef-path=/local/workspace/tappas/apps/gstreamer/general/detection/resources/yolov5m_wo_spp_60p.hef batch-size=1 ! \ queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \ hailofilter function-name=yolov5 so-path=/local/workspace/tappas/apps/gstreamer/libs/post_processes//libyolo_post.so \ config-path=/local/workspace/tappas/apps/gstreamer/general/detection/resources/configs/yolov5.json qos=false ! \ queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! \ person_detect_agg.sink_1 person_detect_agg. ! \ queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \ hailooverlay qos=false ! \ queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \ videoconvert ! fpsdisplaysink video-sink=xvimagesink name=hailo_display sync=false text-overlay=false`

**pipeline breakdown**

1.  Specifies the location of the video used, then decodes and converts to the required format.
    

`gst-launch-1.0 filesrc location=/local/workspace/tappas/apps/gstreamer/resources/mp4/street_night.mp4 name=src_0 ! \ decodebin ! videoconvert !`

2\. Defines the hailocropper process cropping algorithm

`hailocropper so-path=/local/workspace/tappas/apps/gstreamer/libs/post_processes//cropping_algorithms/libwhole_buffer.so \ use-letterbox=false function-name=create_crops internal-offset=true name=person_detect_cropper \`

3\. Sets the aggregator name

`hailoaggregator name=person_detect_agg`

4\. Defines the first cropper src which passes the original frame and connects to the aggregator

`person_detect_cropper.src_0 ! \ queue name=person_detect_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \ person_detect_agg.sink_0`

5\. Defines the cropper second src, and sets the frames into a queue

`person_detect_cropper.src_1 ! \ queue name=hailo_person_detect_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \`

6\. This section is a bug fix!!!! Due to a bug in the cropper, it fails to convert the video the expected input format of the hailonet, this fix does the expected video convert.

`video/x-raw, pixel-aspect-ratio=1/1,format=NV12 ! \ queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! videoconvert n-threads=2 qos=false ! \ queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \`

7\. Performs the inference on the Hailo-8 device.

`hailonet hef-path=/local/workspace/tappas/apps/gstreamer/general/detection/resources/yolov5m_wo_spp_60p.hef batch-size=1 ! \ queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \`

8\. The hailofilter performs a Yolov5 post-process.

`hailofilter function-name=yolov5 so-path=/local/workspace/tappas/apps/gstreamer/libs/post_processes//libyolo_post.so \ config-path=/local/workspace/tappas/apps/gstreamer/general/detection/resources/configs/yolov5.json qos=false ! \ queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! \`

9\. Connects the hailofilter output to aggregator sink

`person_detect_agg.sink_1 person_detect_agg. ! \ queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \`

10\. Overlays the metadata of all collected crops over the original frame

`hailooverlay qos=false ! \ queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \`

11\. Apply the final convert to let GStreamer utilize the format required by the fpsdisplaysink element

`videoconvert ! fpsdisplaysink video-sink=xvimagesink name=hailo_display sync=false text-overlay=false`

Be the first to add a reaction
