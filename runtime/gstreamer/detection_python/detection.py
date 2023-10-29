import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
import subprocess

# Try to import hailo python module
try:
    import hailo
except ImportError:
    exit("Failed to import hailo python module. Make sure you are in hailo virtual environment.")


# This file is a python version of the detection app.
# It should be used with a python post process function.
# See callback_template.py for an example of a post process function.

network_width = 640
network_height = 640
network_format = "RGB"
video_sink = "xvimagesink"

# If TAPPAS version is 3.26.0 or higher, use the following parameters:
nms_score_threshold=0.3 
nms_iou_threshold=0.45
thresholds_str=f"nms-score-threshold={nms_score_threshold} nms-iou-threshold={nms_iou_threshold} output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
# else (TAPPAS version is 3.25.0)
# thresholds_str=""

def parse_arguments():
    parser = argparse.ArgumentParser(description="Detection App")
    parser.add_argument("--input", "-i", type=str, default="/dev/video0", help="Input source. Can be a file, USB or MIPI camera. Defaults to /dev/video0")
    parser.add_argument("--python-module", "-py", type=str, default="callback_template.py", help="Python module with your callback function")
    parser.add_argument("--show-fps", "-f", action="store_true", help="Print FPS on sink")
    parser.add_argument("--disable-sync", action="store_true", help="Disables display sink sync, will run as fast possible.")
    parser.add_argument("--dump-dot", action="store_true", help="Dump the pipeline graph to a dot file pipeline.dot")
    return parser.parse_args()

def QUEUE(name, max_size_buffers=5, max_size_bytes=0, max_size_time=0):
    return f"queue name={name} max-size-buffers={max_size_buffers} max-size-bytes={max_size_bytes} max-size-time={max_size_time} ! "

def get_source_type(input_source):
    # This function will return the source type based on the input source
    # return values can be "file", "mipi" or "usb"
    if input_source.startswith("/dev/video"):
        try:
            output = subprocess.check_output(['v4l2-ctl', '-d', input_source, '--info']).decode('utf-8')
            if 'usb' in output.lower():
                return 'usb'
            elif 'mipi' in output.lower():
                return 'mipi'
            elif 'bcm2835' in output.lower():
                return 'mipi'
            else:
                exit(f"Unknown source type: {output}")
        except Exception as e:
            print("Failed to get camera type, install v4l-utils to support. Assuming usb camera")
            return 'usb'
    else:
        return 'file'
  

class GStreamerApp:
    def __init__(self, args):
        # Create an empty options menu
        self.options_menu = args
        
        # Initialize variables
        tappas_workspace = os.environ.get('TAPPAS_WORKSPACE', '')
        if tappas_workspace == '':
            print("TAPPAS_WORKSPACE environment variable is not set. Please set it to the path of the TAPPAS workspace.")
            exit(1)
        self.current_path = os.getcwd()
        self.postprocess_dir = os.path.join(tappas_workspace, 'apps/h8/gstreamer/libs/post_processes')
        self.default_postprocess_so = os.path.join(self.postprocess_dir, 'libyolo_hailortpp_post.so')
        self.default_network_name = "yolov5"
        self.video_source = self.options_menu.input
        self.source_type = get_source_type(self.video_source)
        self.hef_path = os.path.join(tappas_workspace, 'apps/h8/gstreamer/resources/hef/yolov5m_wo_spp_60p.hef')
        
        if (self.options_menu.disable_sync):
            self.sync = "false" 
        else:
            self.sync = "true"
        
        if (self.options_menu.dump_dot):
            os.environ["GST_DEBUG_DUMP_DOT_DIR"] = self.current_path
        
        # Initialize GStreamer
        
        Gst.init(None)
        
        # Create a GStreamer pipeline 
        self.pipeline = self.create_pipeline()
        
        # connect to hailo_display fps-measurements
        if (self.options_menu.show_fps):
            print("Showing FPS")
            self.pipeline.get_by_name("hailo_display").connect("fps-measurements", self.on_fps_measurement)

        # Create a GLib Main Loop
        self.loop = GLib.MainLoop()
    
    def on_fps_measurement(self, sink, fps, droprate, avgfps):
        print(f"FPS: {fps:.2f}, Droprate: {droprate:.2f}, Avg FPS: {avgfps:.2f}")
        return True

    def create_pipeline(self):
        pipeline_string = self.get_pipeline_string()
        try:
            pipeline = Gst.parse_launch(pipeline_string)
        except Exception as e:
            print(e)
            print(pipeline_string)
            exit(1)
        return pipeline

    def bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End-of-stream")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}")
            loop.quit()
        return True
    
    
    def get_pipeline_string(self):
        if (self.source_type == "mipi"):
            source_element = f"v4l2src device={self.video_source} name=src_0 ! "
            source_element += f"video/x-raw, format={network_format}, width={network_width}, height={network_height}, framerate=30/1 ! "
        
        elif (self.source_type == "usb"):
            source_element = f"v4l2src device={self.video_source} name=src_0 ! "
            source_element += f"video/x-raw, width=640, height=480, framerate=30/1 ! "
        else:  
            source_element = f"filesrc location={self.video_source} name=src_0 ! "
            source_element += QUEUE("queue_dec264", max_size_buffers=5)
            source_element += f" qtdemux ! h264parse ! avdec_h264 max-threads=2 ! "
            source_element += f" video/x-raw,format=I420 ! "
        source_element += QUEUE("queue_scale", max_size_buffers=5)
        source_element += f" videoscale n-threads=2 ! "
        source_element += QUEUE("queue_src_convert", max_size_buffers=5)
        source_element += f" videoconvert n-threads=3 name=src_convert ! "
        source_element += f"video/x-raw, format={network_format}, width={network_width}, height={network_height}, pixel-aspect-ratio=1/1 ! "
        
        pipeline_string = source_element
        pipeline_string += QUEUE("queue_hailonet", max_size_buffers=5)
        pipeline_string += f"hailonet hef-path={self.hef_path} batch-size=1 {thresholds_str} ! "
        pipeline_string += QUEUE("queue_hailofilter", max_size_buffers=5)
        pipeline_string += f"hailofilter function-name={self.default_network_name} so-path={self.default_postprocess_so} qos=false ! "
        pipeline_string += QUEUE("queue_hailopython", max_size_buffers=5)
        pipeline_string += f"hailopython qos=false module={self.options_menu.python_module} ! "
        pipeline_string += QUEUE("queue_hailooverlay", max_size_buffers=5)
        pipeline_string += f"hailooverlay ! "
        pipeline_string += QUEUE("queue_videoconvert", max_size_buffers=5)
        pipeline_string += f"videoconvert n-threads=3 ! "
        pipeline_string += f"fpsdisplaysink video-sink={video_sink} name=hailo_display sync={self.sync} text-overlay={self.options_menu.show_fps} signal-fps-measurements=true"
        return pipeline_string
    
    def dump_dot_file(self):
        print("Dumping dot file...")
        Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.ALL, "pipeline")
        return False
    
    def run(self):
        # Add a watch for messages on the pipeline's bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)

        # Set pipeline to PLAYING state
        self.pipeline.set_state(Gst.State.PLAYING)
        
        # dump dot file
        if (self.options_menu.dump_dot):
            GLib.timeout_add_seconds(3, self.dump_dot_file)
            
        # Run the GLib event loop
        try:
            self.loop.run()
        except:
            pass

        # Clean up
        self.pipeline.set_state(Gst.State.NULL)

# Example usage
if __name__ == "__main__":
    args = parse_arguments()
    app = GStreamerApp(args)
    app.run()
