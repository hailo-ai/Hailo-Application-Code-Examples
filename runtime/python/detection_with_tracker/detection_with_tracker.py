#!/usr/bin/env python3
"""Example module for Hailo Detection + ByteTrack + Supervision."""

import argparse
import supervision as sv
import numpy as np
from tqdm import tqdm
import cv2

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoInference

def initialize_arg_parser():
    """Initialize argument parser for the script."""
    parser = argparse.ArgumentParser(description="Detection Example - Tracker with ByteTrack and Supervision")
    parser.add_argument("-n", "--net", help="Path for the HEF model.", default="yolov5m_wo_spp_60p.hef")
    parser.add_argument("-i", "--input_video", default="input_video.mp4", help="Path to the input video.")
    parser.add_argument("-o", "--output_video", default="output_video.mp4", help="Path to the output video.")
    parser.add_argument("-l", "--labels", default="coco.txt", help="Path to a text file containing labels.")
    parser.add_argument("-s", "--score_thresh", type=float, default=0.5, help="Score threshold - between 0 and 1.")
    return parser

def extract_detections(hailo_output, h, w, threshold=0.5):
    """Extract detections from the HailoRT-postprocess output."""
    xyxy = []
    confidence = []
    class_id = []
    num_detections = 0

    for i, detections in enumerate(hailo_output):
        if len(detections) == 0:
            continue
        for detection in detections:
            bbox = detection[:4]
            # Convert bbox to xyxy absolute pixel values
            bbox[0], bbox[1], bbox[2], bbox[3] = bbox[1] * w, bbox[0] * h, bbox[3] * w, bbox[2] * h
            score = detection[4]

            if score < threshold:
                continue

            xyxy.append(bbox)
            confidence.append(score)
            class_id.append(i)
            num_detections = num_detections + 1

    return {'xyxy': np.array(xyxy),
            'confidence': np.array(confidence), 
            'class_id': np.array(class_id),
            'num_detections': num_detections}

if __name__ == "__main__":
    # Parse command-line arguments
    args = initialize_arg_parser().parse_args()

    hailo_inference = HailoInference(args.net)
    model_h, model_w, _ = hailo_inference.get_input_shape()

    # Initialize components for video processing
    frame_generator = sv.get_video_frames_generator(source_path=args.input_video)
    video_info = sv.VideoInfo.from_video_path(video_path=args.input_video)
    video_w, video_h = video_info.resolution_wh
    box_annotator = sv.RoundBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    tracker = sv.ByteTrack()
    start, end = sv.Point(x=0, y=1080), sv.Point(x=3840, y=1080)
    line_zone = sv.LineZone(start=start, end=end)

    # Load class names from the labels file
    with open(args.labels, 'r', encoding="utf-8") as f:
        class_names = f.read().splitlines()

    # Initialize video sink for output
    with sv.VideoSink(target_path=args.output_video, video_info=video_info) as sink:
        # Process each frame in the video
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            # Resize frame if needed to match model input size
            if model_h != video_h or model_w != video_w:
                preprocessed_frame = cv2.resize(frame,(model_w,model_h))

            # Run inference on the preprocessed frame
            results = hailo_inference.run(preprocessed_frame)

            # Extract detections from the inference results
            detections = extract_detections(results[0], video_h, video_w, args.score_thresh)

            sv_detections = sv.Detections(xyxy=detections['xyxy'],
                                         confidence=detections['confidence'],
                                         class_id=detections['class_id'])

            # Update detections with tracking information
            sv_detections = tracker.update_with_detections(sv_detections)

            # Generate tracked labels for annotated objects
            labels = [
                f"#{tracker_id} {class_names[class_id]}"
                for class_id, tracker_id
                in zip(sv_detections.class_id, sv_detections.tracker_id)
            ]

            # Annotate objects with bounding boxes
            annotated_frame = box_annotator.annotate(
                scene=frame.copy(), detections=sv_detections
            )
            # Annotate objects with labels
            annotated_labeled_frame = label_annotator.annotate(
                scene=annotated_frame, detections=sv_detections, labels=labels
            )
            # Write annotated frame to output video
            sink.write_frame(frame=annotated_labeled_frame)
