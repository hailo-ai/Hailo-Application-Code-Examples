import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import hailo

class app_callback_class:
    def __init__(self):
        self.frame_count = 0
        self.use_frame = False
        # self.frame_queue = multiprocessing.Queue(maxsize=3)
        self.running = True

    def increment(self):
        self.frame_count += 1

    def get_count(self):
        return self.frame_count 

    # def set_frame(self, frame):
    #     if not self.frame_queue.full():
    #         self.frame_queue.put(frame)
        
    # def get_frame(self):
    #     if not self.frame_queue.empty():
    #         return self.frame_queue.get()
    #     else:
    #         return None


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
        classifications = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)
        for classification in classifications:
            label = classification.get_label()
            confidence = classification.get_confidence()
            string_to_print += f"CLIP Classification: {label} {confidence:.2f}\n"
        if isinstance(detection, hailo.HailoDetection):
            label = detection.get_label()
            bbox = detection.get_bbox()
            confidence = detection.get_confidence()
            string_to_print += f"Detection: {label} {confidence:.2f}\n"
            
    print(string_to_print)
    return Gst.PadProbeReturn.OK
    