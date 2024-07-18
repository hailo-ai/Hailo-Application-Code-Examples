import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import hailo

class app_callback_class:
    def __init__(self):
        self.frame_count = 0
        self.use_frame = False
        self.running = True

    def increment(self):
        self.frame_count += 1

    def get_count(self):
        return self.frame_count


def app_callback(self, pad, info, user_data):
    """
    This is the callback function that will be called when data is available
    from the pipeline.
    Processing time should be kept to a minimum in this function.
    If longer processing is needed, consider using a separate thread / process.
    """
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK
    string_to_print = ""
    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    if len(detections) == 0:
        detections = [roi] # Use the ROI as the detection
    # Parse the detections
    for detection in detections:
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        track_id = None
        label = None
        confidence = 0.0
        for track_id_obj in track:
            track_id = track_id_obj.get_id()
        if track_id is not None:
            string_to_print += f'Track ID: {track_id} '
        classifications = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)
        if len(classifications) > 0:
            string_to_print += ' CLIP Classifications:'
            for classification in classifications:
                label = classification.get_label()
                confidence = classification.get_confidence()
                string_to_print += f'Label: {label} Confidence: {confidence:.2f} '
            string_to_print += '\n'
        if isinstance(detection, hailo.HailoDetection):
            label = detection.get_label()
            bbox = detection.get_bbox()
            confidence = detection.get_confidence()
            string_to_print += f"Detection: {label} {confidence:.2f}\n"
    if string_to_print:
        print(string_to_print)
    return Gst.PadProbeReturn.OK
