import hailo
import time
# Importing VideoFrame before importing GST is must
from gsthailo import VideoFrame
from gi.repository import Gst

def map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def aspect_ratio_fix(video_frame: VideoFrame, aspect_ratio=16/9, square_bbox=False):
    # scale the bbox to fit original aspect ratio
    # the original aspect ratio is 16:9 for example
    # the inferred image aspect ratio is 1:1 with borders on the top and bottom
    #|----------------------|
    #|    (black border     |
    #|                      |
    #|------top_border------|
    #|                      |
    #|     scaled image     |
    #|                      |
    #|----bottom_border-----|
    #|                      |
    #|    (black border     |
    #|----------------------|   
    bottom_border = (1-1/aspect_ratio)/2
    top_border = 1 - bottom_border
    detections = video_frame.roi.get_objects_typed(hailo.HAILO_DETECTION)
    for detection in detections:
        bbox = detection.get_bbox()
        # lets map y coordinates to the original image
        ymin = map(bbox.ymin(), bottom_border, top_border, 0, 1)
        ymax = map(bbox.ymax(), bottom_border, top_border, 0, 1)
        height = ymax - ymin
        # get required x coordinates
        xmin = bbox.xmin()  
        width = bbox.width()
        
        if (square_bbox):
            # in addition we want to get square bboxes to prevent distorsion in the cropper
            # lets get make the bbox square (need to take aspect ratio into account)
            normalized_height = height / aspect_ratio
            if normalized_height > width:
                xmin = xmin + (width - height / aspect_ratio)/2
                width = height / aspect_ratio
            elif normalized_height < width:
                ymin = ymin + (height - width * aspect_ratio)/2
                height = width * aspect_ratio
        new_bbox = hailo.HailoBBox(xmin, ymin, width, height)
        detection.set_bbox(new_bbox)

    return Gst.FlowReturn.OK

def run(video_frame: VideoFrame):
    # This function is augmenting the detections to fit the original aspect ratio of the image.
    # import ipdb; ipdb.set_trace()
    roi = video_frame.roi
    roi_bbox = hailo.create_flattened_bbox(roi.get_bbox(), roi.get_scaling_bbox())
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    for detection in detections:
        detection_bbox = detection.get_bbox()
        xmin = (detection_bbox.xmin() * roi_bbox.width()) + roi_bbox.xmin()
        ymin = (detection_bbox.ymin() * roi_bbox.height()) + roi_bbox.ymin()
        xmax = (detection_bbox.xmax() * roi_bbox.width()) + roi_bbox.xmin()
        ymax = (detection_bbox.ymax() * roi_bbox.height()) + roi_bbox.ymin()

        new_bbox = hailo.HailoBBox(xmin, ymin, xmax - xmin, ymax - ymin)
        detection.set_bbox(new_bbox)

    # Clear the scaling bbox of main roi because all detections are fixed.
    new_scaling_bbox = hailo.HailoBBox(0, 0, 1, 1)
    roi.set_scaling_bbox(new_scaling_bbox)
    return Gst.FlowReturn.OK

def fix_16_9(video_frame: VideoFrame):
    return aspect_ratio_fix(video_frame, 16/9)

def fix_4_3(video_frame: VideoFrame):
    return aspect_ratio_fix(video_frame, 4/3)

def fix_16_9_square(video_frame: VideoFrame):
    return aspect_ratio_fix(video_frame, 16/9, True)

def fix_4_3_square(video_frame: VideoFrame):
    return aspect_ratio_fix(video_frame, 4/3, True)
