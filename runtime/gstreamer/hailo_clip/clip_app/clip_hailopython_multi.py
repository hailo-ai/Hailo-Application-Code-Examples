import json
from pathlib import Path

import hailo
import numpy as np
from collections import deque

# Importing VideoFrame before importing GST is must
from gsthailo import VideoFrame
from gi.repository import Gst
# import text_image_matcher instance to make sure that only one instance of the TextImageMatcher class is created.         
from clip_app.TextImageMatcher import text_image_matcher

class StreamDataCollector:
    def __init__(self):
        self.data_by_stream = {}  # Stores data for each stream
        self.stream_queue = deque()  # Tracks the order of incoming streams

    def callback(self, stream_id, data):
        if stream_id in self.stream_queue:
            self.run_computation()
            self.stream_queue.clear()

        self.stream_queue.append(stream_id)
        self.data_by_stream[stream_id] = data

    def run_computation(self):
        # Run your computation here using data in self.data_by_stream
        # print("Running computation with data:", self.data_by_stream)
        embeddings_np = None
        used_detection = []
        for stream_id, detections in self.data_by_stream.items():
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
            best_match_similarity = None
            best_match_stream_id = None
            for match in matches:
                # (row_idx, label, confidence, entry_index) = match
                detection = used_detection[match.row_idx]
                old_classification = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)
                for old in old_classification:
                    detection.remove_object(old)
                if (match.negative or not match.passed_threshold):
                    continue # Don't add classification
                # Add label as classification metadata
                classification = hailo.HailoClassification('clip', match.text, match.similarity)
                detection.add_object(classification)
                # print(f'Best match in stream {detection.get_stream_id()} is {match.text} with similarity {match.similarity}')
                if (best_match_similarity is None or match.similarity > best_match_similarity):
                    best_match_similarity = match.similarity
                    best_match_stream_id = detection.get_stream_id()
            if (best_match_stream_id is not None):
                text_image_matcher.user_data = best_match_stream_id

collector = StreamDataCollector()

def run(video_frame: VideoFrame):
    stream_id = video_frame.roi.get_stream_id()
    top_level_matrix = video_frame.roi.get_objects_typed(hailo.HAILO_MATRIX)
    if len(top_level_matrix) == 0:
        detections = video_frame.roi.get_objects_typed(hailo.HAILO_DETECTION)
    else:
        detections = [video_frame.roi] # Use the ROI as the detection
    # send detections to the collector if all streams have been received for this frame match will be called
    collector.callback(stream_id, detections)
    return Gst.FlowReturn.OK