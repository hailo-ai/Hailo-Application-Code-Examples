import json
from pathlib import Path

import hailo
import numpy as np

# Importing VideoFrame before importing GST is must
from gsthailo import VideoFrame
from gi.repository import Gst

from clip_app.TextImageMatcher import text_image_matcher

def run(video_frame: VideoFrame):
    top_level_matrix = video_frame.roi.get_objects_typed(hailo.HAILO_MATRIX)
    if len(top_level_matrix) == 0:
        detections = video_frame.roi.get_objects_typed(hailo.HAILO_DETECTION)
    else:
        detections = [video_frame.roi] # Use the ROI as the detection
    
    embeddings_np = None
    used_detection = []
    for detection in detections:
        results = detection.get_objects_typed(hailo.HAILO_MATRIX)
        if len(results) == 0:
            # print("No matrix found in detection")
            continue
        # Convert the matrix to a NumPy array
        detection_embedings = np.array(results[0].get_data())
        used_detection.append(detection)
        if embeddings_np is None:
            embeddings_np = detection_embedings[np.newaxis, :]
        else:
            embeddings_np = np.vstack((embeddings_np, detection_embedings))

    if embeddings_np is not None:
        matches = text_image_matcher.match(embeddings_np, report_all=True)
        for match in matches:
            # (row_idx, label, confidence, entry_index) = match
            detection = used_detection[match.row_idx]
            old_classification = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)
            for old in old_classification:
                detection.remove_object(old)
            if (match.negative or not match.passed_threshold):
                continue # Don't add classification just remove the old one  
            # Add label as classification metadata
            classification = hailo.HailoClassification('clip', match.text, match.similarity)
            detection.add_object(classification)
    return Gst.FlowReturn.OK
