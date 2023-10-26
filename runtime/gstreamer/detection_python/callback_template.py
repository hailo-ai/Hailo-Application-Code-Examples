# This is a template for a callback function that can be used with hailo detection.py
# Note that you must be in hailo virtual environment to run use these imports
import hailo
# Importing VideoFrame before importing GST is must
from gsthailo import VideoFrame
from gi.repository import Gst
import numpy as np
import sys

# This is the callback function that will be called for each frame
def run(video_frame: VideoFrame):
    global frame
    # get the detections from the frame
    detections = video_frame.roi.get_objects_typed(hailo.HAILO_DETECTION)
    if (detections is None) or (len(detections) == 0):
        return Gst.FlowReturn.OK
    
    # Get the video info from the video frame
    width = video_frame.video_info.width
    height = video_frame.video_info.height

    # Get the numpy array from the video frame
    with video_frame.map_buffer() as map_info:
        # Create a NumPy array from the buffer data
        numpy_frame_ro = VideoFrame.numpy_array_from_buffer(map_info, video_info=video_frame.video_info)
        # Note this is a read only copy of the frame
        # If you want to modify the frame you need to make a copy
        # numpy_frame = numpy_frame_ro.copy()
        # Note the modifing the frame will not change the original frame (for this you'll need to replave the buffer data)
    # parse the detections
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        if label == "person":
            string_to_print = (f"Detection: {label} {confidence}")
            sys.stdout.write(string_to_print)
            sys.stdout.write("\r")
            sys.stdout.flush()
    return Gst.FlowReturn.OK

# This function will be called when the pipeline is closed
def close():
    print("Python close function called")

