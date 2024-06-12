#!/usr/bin/env python3
import numpy as np
from loguru import logger
import argparse
import cv2
import time
from ufld_utils import UFLDProcessing

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoInference

def parser_init():
    """
    Initialize and configure the argument parser for this script.
    
    Returns:
        argparse.ArgumentParser: An instance of ArgumentParser.
    """
    parser = argparse.ArgumentParser(description="UFLD_v2 inference")

    parser.add_argument(
        "-m",
        "--model",
        help="Path of ufld_v2.hef",
        default="ufld_v2.hef"
    )

    parser.add_argument(
        "-i",
        "--input_video",
        default="input_video.mp4",
        help="Path of the video to perform inference on.",
    )
    
    parser.add_argument(
        "-o",
        "--output_video",
        default="output_video.mp4",
        help="Path of the output video.",
    )

    return parser


if __name__ == "__main__":

    # Parse command-line arguments
    args = parser_init().parse_args()
    
    # Initialize HailoInference with the specified model
    hailo_inference = HailoInference(args.model)
    
     # Get input shape information 
    input_height, input_width, _ = hailo_inference.get_input_shape()
    
    # Initialize UFLDProcessing with the tusimple parameters
    ufld_processing = UFLDProcessing(input_height = input_height,
                                    input_width = input_width,
                                    num_cell_row=100,
                                    num_cell_col=100,
                                    num_row=56,
                                    num_col=41,
                                    num_lanes=4,
                                    crop_ratio=0.8,
                                    )
    
    # Open the input video file for reading
    vidcap = cv2.VideoCapture(args.input_video)
    
    # Configure the output video codec and create an output video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(args.output_video,fourcc,20,(1280,720))
    
     # Get the total number of frames in the input video
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cur_frame = 0
    total_inference_time = 0
    success, frame = vidcap.read()
    while success:
        cur_frame += 1
        print(f"Processing frame {cur_frame}/{frame_count}", end="\r")
        
        # Preprocess the frame for inference
        preprocessed_frame = ufld_processing.resize(frame)
        
        # Perform inference on the preprocessed frame
        start_time = time.time()
        inference_output = hailo_inference.run(preprocessed_frame)
        end_time = time.time()
        
        total_inference_time += (end_time - start_time)
        
        # Extract and visualize lane coordinates from the inference output
        lanes = ufld_processing.get_coordinates(np.array([inference_output[0]]))

        for lane in lanes:
            for coord in lane:
                cv2.circle(frame, coord, 5, (0, 255, 0), -1)

        # Write the frame to the output video
        output_video.write(frame.astype('uint8'))
        
        # Check for user input to exit the processing loop (press 'q' key)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Read the next frame from the input video
        success, frame = vidcap.read()
    
     # Calculate and log the total inference time and frames per second (FPS)
    logger.info("Total inference time: {:.2f} sec, FPS: {:.2f}", total_inference_time, cur_frame/total_inference_time)
    
    # Release the Hailo device
    hailo_inference.release_device()
