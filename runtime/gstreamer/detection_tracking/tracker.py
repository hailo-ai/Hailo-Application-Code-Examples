"""Example module for a GSTreamer pipeline performing Hailo Detection + ByteTrack + Supervision."""


import hailo
from gsthailo import VideoFrame
from gi.repository import Gst
import supervision as sv
import numpy as np

def extract_detections(video_frame, hailo_output, threshold=0.5):
    """Covnert Hailo detections to sv format"""
    xyxy = []
    confidence = []
    class_id = []
    num_detections = 0

    for detection in hailo_output:
        video_frame.roi.remove_object(detection)
        score = detection.get_confidence()
        if score < threshold:
            continue
        bbox = [detection.get_bbox().xmin(), detection.get_bbox().ymin(), detection.get_bbox().xmax(), detection.get_bbox().ymax()]

        xyxy.append(bbox)
        confidence.append(score)
        class_id.append(detection.get_class_id())
        num_detections = num_detections + 1

    return {'xyxy':       np.array(xyxy),
            'confidence': np.array(confidence), 
            'class_id':   np.array(class_id),
            'num_detections': num_detections}

tracker = sv.ByteTrack()

def run(video_frame: VideoFrame):
    """Process every incoming frame, and its detections"""
    detections = hailo.get_hailo_detections(video_frame.roi)

    detections_to_track = extract_detections(video_frame, detections)

    sv_detections = sv.Detections(xyxy=detections_to_track['xyxy'],
                                  confidence=detections_to_track['confidence'],
                                  class_id=detections_to_track['class_id'])

    sv_detections = tracker.update_with_detections(sv_detections)
    for target in sv_detections:
        bbox = hailo.HailoBBox(target[0][0], target[0][1], target[0][2]-target[0][0], target[0][3]-target[0][1])

        if target[2] > 1.0:
            target[2] /= 100
        hailo.add_detection(video_frame.roi, bbox, str(target[4]), target[2], 0)

    return Gst.FlowReturn.OK

