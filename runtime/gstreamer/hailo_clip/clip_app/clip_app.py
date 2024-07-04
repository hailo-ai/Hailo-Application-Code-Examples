import os
import argparse
import logging

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Gst', '1.0')
from gi.repository import Gtk, Gst, GLib

from clip_app.logger_setup import setup_logger, set_log_level
from clip_app.clip_pipeline import get_pipeline
# import text_image_matcher instance to make sure that only one instance of the TextImageMatcher class is created.
from clip_app.TextImageMatcher import text_image_matcher
import clip_app.gui as gui
# Disabling the Accessibility Bus (sends warnings due to docker user issues)
os.environ['NO_AT_BRIDGE'] = '1'

# add logging
logger = setup_logger()
set_log_level(logger, logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Hailo online clip app")
    parser.add_argument("--input", "-i", type=str, default="/dev/video0", help="URI of the input stream.")
    parser.add_argument("--clip-runtime", action="store_true", help="When set app will use clip pytorch runtime for text embedding.")
    parser.add_argument("--detector", "-d", type=str, choices=["person", "face", "none"], default="none", help="Which detection pipeline to use.")
    parser.add_argument("--json-path", type=str, default=None, help="Path to json file to load and save embeddings. If not set embeddings.json will be used.")
    parser.add_argument("--sync", action="store_true", help="Enable display sink sync.")
    parser.add_argument("--dump-dot", action="store_true", help="Dump the pipeline graph to a dot file.")
    parser.add_argument("--detection-threshold", type=float, default=0.5, help="Detection threshold")
    return parser.parse_args()


def on_destroy(window):
    print("Destroying window...")
    window.quit_button_clicked(None)


def main():
    args = parse_arguments()
    win = AppWindow(args)
    
    win.connect("destroy", on_destroy)
    win.show_all()
    Gtk.main()


# TEMP
import hailo


def buffer_probe(pad, info):
    buffer = info.get_buffer()
    if buffer:
        roi = hailo.get_roi_from_buffer(buffer)
        # import ipdb; ipdb.set_trace()
        print(f'buffer_probe {roi.get_stream_id()}')
    return Gst.PadProbeReturn.OK


class AppWindow(Gtk.Window):
    # Add GUI functions to the AppWindow class
    build_ui = gui.build_ui
    add_text_boxes = gui.add_text_boxes
    update_text_boxes = gui.update_text_boxes
    update_text_prefix = gui.update_text_prefix
    quit_button_clicked = gui.quit_button_clicked
    on_text_box_updated = gui.on_text_box_updated
    on_slider_value_changed = gui.on_slider_value_changed
    on_negative_check_button_toggled = gui.on_negative_check_button_toggled
    on_ensemble_check_button_toggled = gui.on_ensemble_check_button_toggled
    on_load_button_clicked = gui.on_load_button_clicked
    on_save_button_clicked = gui.on_save_button_clicked
    update_progress_bars = gui.update_progress_bars
    on_track_id_update = gui.on_track_id_update

    def __init__(self, args):
        Gtk.Window.__init__(self, title="Clip App")
        self.set_border_width(10)
        self.set_default_size(400, 200)
        
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        # move self.current_path one directory up to get the path to the workspace
        self.current_path = os.path.dirname(self.current_path)
        os.environ["GST_DEBUG_DUMP_DOT_DIR"] = self.current_path
        
        self.tappas_postprocess_dir = os.environ.get('TAPPAS_POST_PROC_DIR', '')
        if self.tappas_postprocess_dir == '':
            print("TAPPAS_POST_PROC_DIR environment variable is not set. Please set it by sourcing setup_env.sh")
            exit(1)
        
        self.input_uri = args.input
        self.dump_dot = args.dump_dot
        self.sync = 'true' if args.sync else 'false'
        self.json_file = os.path.join(self.current_path, "embeddings.json") if args.json_path is None else args.json_path
        self.use_default_text = args.json_path is None
        self.detector = args.detector

        # get current path
        Gst.init(None)
        self.pipeline = self.create_pipeline()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)

        # get xvimagesink element and disable qos
        # xvimagesink is instantiated by fpsdisplaysink
        hailo_display = self.pipeline.get_by_name("hailo_display")
        xvimagesink = hailo_display.get_by_name("xvimagesink0")
        xvimagesink.set_property("qos", False)
        
        # get text_image_matcher instance
        self.text_image_matcher = text_image_matcher
        self.text_image_matcher.set_threshold(args.detection_threshold)

        # TEMP
        # # connect probe to element to parse buffer
        # queue = self.pipeline.get_by_name("queue27")
        # # Attach the buffer probe to the queue's src pad
        # queue_src_pad = queue.get_static_pad("src")
        # if not queue_src_pad:
        #     print("Unable to get the queue's src pad.")
        #     return
        # queue_src_pad.add_probe(Gst.PadProbeType.BUFFER, buffer_probe)

        # build UI
        self.build_ui(args)
        
        # set runtime
        if args.clip_runtime:
            self.text_image_matcher.init_clip()
        else:
            print(f"No text embedding runtime selected, adding new text is disabled. Loading {self.json_file}")
            for text_box in self.text_boxes:
                text_box.set_editable(False)
            self.on_load_button_clicked(None)
        
        if self.text_image_matcher.model_runtime is not None:
            print(f"Using {self.text_image_matcher.model_runtime} for text embedding")
            if not self.use_default_text:
                self.on_load_button_clicked(None)
            else:
                print("Adding some default text entries. To disable this use --json-path to load from JSON file.")
                self.add_default_texts()
        
        # start the pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
         
        if self.dump_dot:
            GLib.timeout_add_seconds(5, self.dump_dot_file)
        
        self.update_text_boxes()

        # Define a timeout duration in nanoseconds (e.g., 5 seconds)
        timeout_ns = 5 * Gst.SECOND

        # Wait until state change is done or until the timeout occurs
        state_change_return, state, pending = self.pipeline.get_state(timeout_ns)

        if state_change_return == Gst.StateChangeReturn.SUCCESS:
            print("Pipeline state changed to PLAYING successfully.")
        elif state_change_return == Gst.StateChangeReturn.ASYNC:
            print("State change is ongoing asynchronously.")
        elif state_change_return == Gst.StateChangeReturn.FAILURE:
            print("State change failed.")
        else:
            print("Unknown state change return value.")

    



    def add_default_texts(self):
        if self.detector == "person":
            self.text_image_matcher.add_text("person", 0, True)
            self.text_image_matcher.add_text("person with a water bottle", 1)
            self.text_image_matcher.add_text("person with a hat", 2)
            self.text_image_matcher.add_text("man with a bag", 3)
            self.text_image_matcher.add_text("woman with glasses", 4)
        elif self.detector == "face":
            self.text_image_matcher.add_text("face", 0, True)
            self.text_image_matcher.add_text("smiling face", 1)
            self.text_image_matcher.add_text("person raising eyebrows", 2)
            self.text_image_matcher.add_text("person winking their eye", 3)
        elif self.detector == "fast_sam":
            self.text_image_matcher.add_text("object", 0, True)
            self.text_image_matcher.add_text("cell phone", 1)
            self.text_image_matcher.add_text("person", 2)
            self.text_image_matcher.add_text("car", 3)
            self.text_image_matcher.add_text("table", 4)
            self.text_image_matcher.add_text("computer", 5)
        elif self.detector == "none":
            self.text_image_matcher.add_text("empty room", 0, True)
            self.text_image_matcher.add_text("person", 1)
            self.text_image_matcher.add_text("cellphone", 2)


    def dump_dot_file(self):
        print("Dumping dot file...")
        Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.ALL, "pipeline")
        return False


    def on_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            self.shutdown()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print("Error: %s" % err, debug)
            self.shutdown()
        # print QOS messages
        elif t == Gst.MessageType.QOS:
            # print which element is reporting QOS
            src = message.src.get_name()
            print(f"QOS from {src}")
        return True


    def shutdown(self):
        self.pipeline.set_state(Gst.State.NULL)
        Gtk.main_quit()


    def create_pipeline(self):
        pipeline_str = get_pipeline(self.current_path, self.detector, self.sync, self.input_uri, self.tappas_postprocess_dir)
        print(f'PIPELINE:\ngst-launch-1.0 {pipeline_str}')
        try:
            pipeline = Gst.parse_launch(pipeline_str)
        except GLib.Error as e:
            logger.error(f"An error occurred while parsing the pipeline: {e}")
        return pipeline


if __name__ == "__main__":
    main()
