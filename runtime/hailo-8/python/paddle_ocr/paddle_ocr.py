#!/usr/bin/env python3
import argparse
import os
import sys
from loguru import logger
import queue
import threading
from functools import partial
import time
from paddle_ocr_utils import det_postprocess, resize_with_padding, inference_result_handler, OcrCorrector, map_bbox_to_original_image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.hailo_inference import HailoInfer
from common.toolbox import init_input_source, preprocess, visualize, FrameRateTracker
import uuid
from collections import defaultdict

# A dictionary that accumulates all OCR crops and their results for a single frame.
ocr_results_dict = defaultdict(lambda: {"frame": None, "results": [], "boxes": [], "count": 0})
ocr_expected_counts = {}


def parse_args() -> argparse.Namespace:
    """
    Initialize argument parser for the script.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Paddle OCR Example with detection + OCR networks")

    parser.add_argument(
        "-n", "--net",
        help="Space-separated paths for the networks in HEF format, e.g.: ocr_det.hef ocr.hef",
        nargs="+",
        required=True
    )
    parser.add_argument(
        "-i", "--input", default="ocr_img1.png",
        help="Path to the input - either an image or a folder of images."
    )
    parser.add_argument(
        "-b", "--batch_size", default=1, type=int, required=False,
        help="Number of images in one batch"
    )
    parser.add_argument(
        "-s", "--save_stream_output", action="store_true",
        help="Save the output of the inference from a stream."
    )
    parser.add_argument(
        "-o", "--output-dir", help="Directory to save the results.",
        default=None
    )
    parser.add_argument(
        "-r", "--resolution",
        choices=["sd", "hd", "fhd"],
        default="sd",
        help="Camera only: Choose input resolution: 'sd' (640x480), 'hd' (1280x720), or 'fhd' (1920x1080). Default is 'sd'."
    )
    parser.add_argument(
        "--show-fps", action="store_true",
        help="Enable FPS performance measurement."
    )

    parser.add_argument(
        "--use-corrector", action="store_true",
        help="Enable text correction after OCR (e.g., for spelling or formatting)."
    )

    args = parser.parse_args()

    if len(args.net) != 2:
        raise ValueError("Please provide exactly two HEF files: det_net hef and ocr_net hef")

    args.det_net, args.ocr_net = args.net

    # Validate HEF paths
    if not os.path.exists(args.det_net):
        raise FileNotFoundError(f"Detector HEF file not found: {args.det_net}")
    if not os.path.exists(args.ocr_net):
        raise FileNotFoundError(f"OCR HEF file not found: {args.ocr_net}")

    # Setup output dir
    if args.output_dir is None:
        args.output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(args.output_dir, exist_ok=True)

    return args



def detector_hailo_infer(hailo_inference, input_queue, output_queue):
    """
    Main inference loop that pulls data from the input queue, runs asynchronous
    inference, and pushes results to the output queue.

    Each item in the input queue is expected to be a tuple:
        (input_batch, preprocessed_batch)
        - input_batch: Original frames (used for visualization or tracking)
        - preprocessed_batch: Model-ready frames (e.g., resized, normalized)

    Args:
        hailo_inference (HailoInfer): The inference engine to run model predictions.
        input_queue (queue.Queue): Provides (input_batch, preprocessed_batch) tuples.
        output_queue (queue.Queue): Collects (input_frame, result) tuples for visualization.

    Returns:
        None
    """
    while True:
        next_batch = input_queue.get()
        if not next_batch:
            break  # Stop signal received

        input_batch, preprocessed_batch = next_batch

        # Prepare the callback for handling the inference result
        inference_callback_fn = partial(
            detector_inference_callback,
            input_batch=input_batch,
            output_queue=output_queue
        )

        # Run async inference
        hailo_inference.run(preprocessed_batch, inference_callback_fn)

    # Release resources and context
    hailo_inference.close()



def ocr_hailo_infer(hailo_inference, input_queue, output_queue):
    """
    Main inference loop that pulls data from the input queue, runs asynchronous
    inference, and pushes results to the output queue.

    Each item in the input queue is expected to be a tuple:
        (input_batch, preprocessed_batch)
        - input_batch: Original frames (used for visualization or tracking)
        - preprocessed_batch: Model-ready frames (e.g., resized, normalized)

    Args:
        hailo_inference (HailoInfer): The inference engine to run model predictions.
        input_queue (queue.Queue): Provides (input_batch, preprocessed_batch) tuples.
        output_queue (queue.Queue): Collects (input_frame, result) tuples for visualization.

    Returns:
        None
    """
    while True:
        next_batch = input_queue.get()
        if not next_batch:
            break  # Stop signal received

        input_batch, preprocessed_batch, extra_context = next_batch

        # Prepare the callback for handling the inference result
        inference_callback_fn = partial(
            ocr_inference_callback,
            input_batch=input_batch,
            output_queue=output_queue,
            extra_context = extra_context
        )

        # Run async inference
        hailo_inference.run(preprocessed_batch, inference_callback_fn)

    # Release resources and context
    hailo_inference.close()


def run_inference_pipeline(det_net, ocr_net, input, batch_size, output_dir,
          save_stream_output=False, resolution="sd", show_fps=False, use_corrector=False) -> None:
    """
    Run full detector + OCR inference pipeline with multi-threading and streaming.

    Args:
        det_net: model path for the detection network.
        ocr_net: model path for the OCR network.
        input (str): Input source — 'camera', image directory, or video file path.
        batch_size (int): Number of frames to process in each batch.
        output_dir (str): Directory where output images or videos will be saved.
        save_stream_output (bool): Whether to save the output stream. Defaults to False.
        resolution (str): Input resolution format (e.g., 'sd', 'hd'). Defaults to 'sd'.
        show_fps (bool): Whether to display frames-per-second performance. Defaults to False.
        use_corrector (bool): Whether to enable text spell correction. Defaults to False.

    Returns:
        None
    """
    # Initialize capture handle for video/camera or load image folder
    cap, images = init_input_source(input, batch_size, resolution)

    # Queues for passing data between threads
    det_input_queue = queue.Queue()
    ocr_input_queue = queue.Queue()

    det_postprocess_queue = queue.Queue()
    ocr_postprocess_queue = queue.Queue()

    vis_output_queue = queue.Queue()


    fps_tracker=None
    if show_fps:
        fps_tracker = FrameRateTracker()

    ocr_corrector = None
    if use_corrector:
        ocr_corrector = OcrCorrector()


    ####### CALLBACKS ########

    # Final visualization callback function with optional correction
    post_process_callback_fn = partial(
        inference_result_handler,
        ocr_corrector=ocr_corrector
    )


    # Detector inference callbacks
    detector_inference_callback_fn = partial(
        detector_inference_callback,
        det_postprocess_queue=det_postprocess_queue,
    )

    # ocr inference callbacks
    ocr_inference_callback_fn = partial(
        ocr_inference_callback,
        ocr_postprocess_queue=ocr_postprocess_queue
    )


    ###### THREADS ########

    # Start detector with async Hailo inference
    detector_hailo_inference = HailoInfer(det_net, batch_size)

    # Start ocr with async Hailo inference
    ocr_hailo_inference = HailoInfer(ocr_net, batch_size, priority=1)

    height, width, _ = detector_hailo_inference.get_input_shape()

    # input postprocess
    preprocess_thread = threading.Thread(
        target=preprocess, args=(images, cap, batch_size, det_input_queue, width, height)
    )

    # detector output postprocess
    detection_postprocess_thread = threading.Thread(
        target=detection_postprocess,
        args=(det_postprocess_queue, ocr_input_queue, vis_output_queue, height, width),
    )

    # ocr output postprocess
    ocr_postprocess_thread = threading.Thread(
        target=ocr_postprocess,
        args=(ocr_postprocess_queue, vis_output_queue),
    )

    # visualisation postprocess
    vis_postprocess_thread = threading.Thread(
        target=visualize, args=(vis_output_queue, cap, save_stream_output,
                                output_dir, post_process_callback_fn, fps_tracker, True)
    )

    det_thread = threading.Thread(
        target=detector_hailo_infer, args=(detector_hailo_inference, det_input_queue, det_postprocess_queue)
    )

    ocr_thread = threading.Thread(
        target=ocr_hailo_infer, args=(ocr_hailo_inference, ocr_input_queue, ocr_postprocess_queue)
    )

    if show_fps:
        fps_tracker.start()

    ##### Start threads ######
    preprocess_thread.start()
    det_thread.start()
    detection_postprocess_thread.start()
    ocr_thread.start()
    ocr_postprocess_thread.start()
    vis_postprocess_thread.start()


    ##### Join Threads and Shutdown Queues ######

    # Wait for input preprocessing to finish
    preprocess_thread.join()

    # Wait for detector inference to finish
    det_thread.join()

    # Tell detection postprocess thread to exit
    det_postprocess_queue.put(None)
    detection_postprocess_thread.join()

    # Signal OCR inference thread to stop (no more crops coming)
    ocr_input_queue.put(None)
    ocr_thread.join()

    # Signal OCR postprocess thread to stop
    ocr_postprocess_queue.put(None)
    ocr_postprocess_thread.join()

    # Signal visualization thread that everything is done
    vis_output_queue.put(None)
    vis_postprocess_thread.join()

    logger.info('Inference was successful!')

    if show_fps:
        logger.debug(fps_tracker.frame_rate_summary())



def detector_inference_callback(
    completion_info,
    bindings_list: list,
    input_batch: list,
    output_queue,
) -> None:
    """
    Callback triggered after detection inference completes.

    Args:
        completion_info: Info about whether inference succeeded or failed.
        bindings_list (list): Output buffer objects for each input.
        input_batch (list): input frames.
        output_queue (queue.Queue): Queue to pass cropped regions to the OCR pipeline.
    Returns:
        None
    """
    if completion_info.exception:
        logger.error(f'Inference error: {completion_info.exception}')
    else:
        for i, bindings in enumerate(bindings_list):
            result = bindings.output().get_buffer()
            output_queue.put(([input_batch[i], result]))



def detection_postprocess(
    det_postprocess_queue: queue.Queue,
    ocr_input_queue: queue.Queue,
    vis_output_queue: queue.Queue,
    model_height,
    model_width,
) -> None:
    """
    Worker thread to handle postprocessing of detection results.

    Args:
        det_postprocess_queue (queue.Queue): Queue containing tuples of (input_frame, preprocessed_img, result).
        ocr_input_queue (queue.Queue): Queue to send cropped and resized regions along with metadata to OCR stage.
        vis_output_queue (queue.Queue): Queue to send empty OCR results directly to visualization if no detections.
        model_height (int): The height of the model input used for scaling detection boxes.
        model_width (int): The width of the model input used for scaling detection boxes.

    Returns:
        None
    """
    while True:
        item = det_postprocess_queue.get()
        if item is None:
            break  # Shutdown signal

        input_frame, result = item

        det_pp_res, boxes = det_postprocess(result, input_frame, model_height, model_width)

        frame_id = str(uuid.uuid4())
        # Register how many OCR crops are expected from this frame
        ocr_expected_counts[frame_id] = len(det_pp_res)

        # If no text regions were detected, skip OCR and go straight to visualization
        if len(det_pp_res) == 0:
            vis_output_queue.put((input_frame, [], []))
            continue

        # For each detected text region:
        for idx, cropped in enumerate(det_pp_res):
            # Resize the cropped region to match OCR input size (with padding)
            resized = resize_with_padding(cropped)
            # Push one OCR task to the OCR input queue
            ocr_input_queue.put((input_frame, [resized], (frame_id, boxes[idx])))



def ocr_inference_callback(
    completion_info,
    bindings_list: list,
    input_batch: list,
    output_queue: queue.Queue,
    extra_context=None
) -> None:
    """
    Callback triggered after OCR inference completes. Extracts the result, attaches metadata,
    and pushes it to the OCR postprocessing queue.

    Args:
        completion_info: Info about whether inference succeeded or failed.
        bindings_list (list): Output buffer objects from the OCR model.
        input_batch (list): input frame (only one image per batch).
        output_queue (queue.Queue): Queue used to send the OCR results and metadata to the postprocessing stage.
        extra_context (tuple, optional): A tuple of (frame_id, [box]), where `box` is the denormalized detection
                                         bounding box from the detector. Used to group OCR results by frame.

    Returns:
        None
    """
    if completion_info.exception:
        logger.error(f"OCR Inference error: {completion_info.exception}")
        return

    # Handle the single result
    result = bindings_list[0].output().get_buffer()

    # Unpack inputs
    original_frame = input_batch
    frame_id, box = extra_context
    output_queue.put((frame_id, original_frame, result, box))


def ocr_postprocess(
    ocr_postprocess_queue: queue.Queue,
    vis_output_queue: queue.Queue
) -> None:
    """
    Worker thread to handle postprocessing of OCR model results.

    Args:
        ocr_postprocess_queue (queue.Queue): Queue containing tuples of (frame_id, input_frame, ocr_output, denorm_box).
        vis_output_queue (queue.Queue): Queue to pass the final results to visualization.

    Returns:
        None
    """
    while True:

        item = ocr_postprocess_queue.get()
        if item is None:
            break  # Shutdown signal

        frame_id, original_frame, ocr_output, denorm_box = item
        ocr_results_dict[frame_id]["results"].append(ocr_output)
        ocr_results_dict[frame_id]["boxes"].append(denorm_box)
        ocr_results_dict[frame_id]["count"] += 1
        ocr_results_dict[frame_id]["frame"] = original_frame

        expected = ocr_expected_counts.get(frame_id, None)

        # If all OCR results for this frame are collected
        if expected is not None and ocr_results_dict[frame_id]["count"] == expected:
            # Push the grouped results to the visualization queue
            vis_output_queue.put((
                ocr_results_dict[frame_id]["frame"],   # The full input frame
                ocr_results_dict[frame_id]["results"], # All OCR outputs for this frame
                ocr_results_dict[frame_id]["boxes"]    # All box positions for this frame
            ))

            # Clean up to free memory
            del ocr_results_dict[frame_id]
            del ocr_expected_counts[frame_id]


def main() -> None:
    """
    Main function to run the script.
    """
    args = parse_args()
    run_inference_pipeline(args.det_net, args.ocr_net, args.input, args.batch_size,
          args.output_dir, args.save_stream_output, args.resolution, args.show_fps, args.use_corrector)


if __name__ == "__main__":
    main()