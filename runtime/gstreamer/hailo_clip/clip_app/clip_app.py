import os
import argparse
import logging

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Gst', '1.0')
from gi.repository import Gtk, Gst, GLib

from clip_app.get_pkg_info import get_pkg_info
from clip_app.logger_setup import setup_logger, set_log_level

from clip_app.clip_pipeline import get_pipeline
from clip_app.clip_pipeline_multi import get_pipeline_multi
# import text_image_matcher instance to make sure that only one instance of the TextImageMatcher class is created.         
from clip_app.TextImageMatcher import text_image_matcher

# Disabling the Accessibility Bus (sends warnings due to docker user issues)
os.environ['NO_AT_BRIDGE'] = '1'

# add logging
logger = setup_logger()
set_log_level(logger, logging.INFO)

# Disable features which are still in development
DISABLE_DEVELOPMENT_FEATURES = True

def parse_arguments():
    parser = argparse.ArgumentParser(description="Hailo online clip app")
    parser.add_argument("--sync", action="store_true", help="Enable display sink sync.")
    parser.add_argument("--input", "-i", type=str, default="/dev/video0", help="URI of the input stream.")
    parser.add_argument("--dump-dot", action="store_true", help="Dump the pipeline graph to a dot file.")
    parser.add_argument("--detection-threshold", type=float, default=0.5, help="Detection threshold")
    if (DISABLE_DEVELOPMENT_FEATURES):
        parser.add_argument("--detector", "-d", type=str, choices=["person", "none"], default="none", help="Which detection pipeline to use.")
    else:
        parser.add_argument("--detector", "-d", type=str, choices=["person", "face", "fast_sam", "none"], default="none", help="Which detection pipeline to use.")
        parser.add_argument("--onnx-runtime", action="store_true", help="When set app will use ONNX runtime for text embedding.")
        parser.add_argument("--multi-stream", action="store_true", help="When set app will use multi stream pipeline. In this mode detector is set to person.")
    parser.add_argument("--clip-runtime", action="store_true", help="When set app will use clip pythoch runtime for text embedding.")
    parser.add_argument("--json-path", type=str, default=None, help="Path to json file to load and save embeddings. If not set embeddings.json will be used. ")
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

class AppWindow(Gtk.Window):
    def __init__(self, args):
        
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        # move self.current_path one directory up to get the path to the workspace
        self.current_path = os.path.dirname(self.current_path)
        os.environ["GST_DEBUG_DUMP_DOT_DIR"] = self.current_path
        
        self.input_uri = args.input
        self.dump_dot = args.dump_dot
        if args.sync:
            self.sync = 'true'
        else:
            self.sync = 'false'
        if args.json_path is None:
            if args.multi_stream:
                self.json_file = os.path.join(self.current_path, "multi_stream_embeddings.json")
            else:
                self.json_file = os.path.join(self.current_path, "embeddings.json")
            self.use_default_text = True
        else:
            self.json_file = args.json_path
            self.use_default_text = False

        self.detector = args.detector
        self.multi_stream = args.multi_stream
        if (self.multi_stream):
            if (self.detector != "person"):
                print("Multi stream mode is enabled. Detector is set to person.")
            self.detector = "person"

        # get TAPPAS version and path
        info = get_pkg_info()
        self.tappas_workspace = info['tappas_workspace']
        self.tappas_version = info['version']

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
        
        # if self.multi_stream:
        #     self.text_image_matcher.run_softmax = False

        # build UI
        self.build_ui(args)
        
        # set video muxing for multi stream
        if self.multi_stream:
            self.input_selector_element = self.pipeline.get_by_name('input_selector')
            self.input_selector_element.set_property('active-pad', self.input_selector_element.get_static_pad('sink_0'))
            # Schedule the update_video_muxing method to be called every second 
            GLib.timeout_add(1000, self.update_video_muxing)

        # set runtime
        if args.onnx_runtime:
            onnx_path = os.path.join(self.current_path, "onnx/textual.onnx")
            self.text_image_matcher.init_onnx(onnx_path)
        elif args.clip_runtime:
            self.text_image_matcher.init_clip()
        else:
            print(f"No text embedding runtime selected, adding new text is disabled. Loading {self.json_file}")
            for text_box in self.text_boxes:
                text_box.set_editable(False)
            self.on_load_button_clicked(None)
        
        if (self.text_image_matcher.model_runtime is not None):
            print(f"Using {self.text_image_matcher.model_runtime} for text embedding")
            if (not self.use_default_text):
                self.on_load_button_clicked(None)   
            else:
                print(f"Adding some default text entries. To disable this use --json-path to load from JSON file. ")
                self.add_default_texts()
        
        # start the pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
         
        if (self.dump_dot):
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

    
    def build_ui(self, args):
        Gtk.Window.__init__(self, title="Clip App")
        self.set_border_width(10)
        self.set_default_size(400, 200)

        ui_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(ui_vbox)
        self.ui_vbox = ui_vbox

        # Slider to control threshold parameter
        # set range to 0.0 - 1.0 with 0.05 increments
        self.slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.0, 1.0, 0.05)
        self.slider.set_value(args.detection_threshold)
        self.slider.connect("value-changed", self.on_slider_value_changed)
        ui_vbox.pack_start(self.slider, False, False, 0)

        # Text boxes to control text embeddings
        self.add_text_boxes()

        # add 2 buttons to hbox load and save
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.load_button = Gtk.Button(label="Load")
        self.load_button.connect("clicked", self.on_load_button_clicked)
        hbox.pack_start(self.load_button, False, False, 0)
        self.save_button = Gtk.Button(label="Save")
        self.save_button.connect("clicked", self.on_save_button_clicked)
        hbox.pack_start(self.save_button, False, False, 0)
        ui_vbox.pack_start(hbox, False, False, 0)


        # Quit Button
        quit_button = Gtk.Button(label="Quit")
        quit_button.connect("clicked", self.quit_button_clicked)
        ui_vbox.pack_start(quit_button, False, False, 0)

    def add_text_boxes(self, N=6):
        """Adds N text boxes to the GUI and sets up callbacks for text changes."""
        self.text_boxes = []
        self.probability_progress_bars = []
        self.negative_check_buttons = []
        self.ensemble_check_buttons = []
        self.text_prefix_labels = []

        # Create vertical boxes for each column
        vbox1 = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        vbox2 = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        vbox3 = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        vbox4 = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        vbox5 = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)

        # Adding header line to the vertical boxes
        vbox1.pack_start(Gtk.Label(label="Negative", width_chars=10), False, False, 0)
        vbox2.pack_start(Gtk.Label(label="Ensemble", width_chars=10), False, False, 0)
        vbox3.pack_start(Gtk.Label(label="Prefix", width_chars=10), False, False, 0)
        vbox4.pack_start(Gtk.Label(label="Text Description", width_chars=20), False, False, 0)
        vbox5.pack_start(Gtk.Label(label="Probability", width_chars=10), False, False, 0)

        for i in range(N):
            # Create and add a negative check button with a callback
            negative_check_button = Gtk.CheckButton()
            negative_check_button.connect("toggled", self.on_negative_check_button_toggled, i)
            vbox1.pack_start(negative_check_button, True, True, 0)
            self.negative_check_buttons.append(negative_check_button)

            # Create and add an ensemble check button
            ensemble_check_button = Gtk.CheckButton()
            ensemble_check_button.connect("toggled", self.on_ensemble_check_button_toggled, i)
            vbox2.pack_start(ensemble_check_button, True, True, 0)
            self.ensemble_check_buttons.append(ensemble_check_button)

            # Create and add a label
            label = Gtk.Label(label=f"{self.text_image_matcher.text_prefix}")
            vbox3.pack_start(label, True, True, 0)
            self.text_prefix_labels.append(label)

            # Create and add a text box with callbacks
            text_box = Gtk.Entry()
            text_box.set_width_chars(20)  # Adjust the width to align with the "Text Description" header
            text_box.connect("activate", lambda widget, idx=i: self.on_text_box_updated(widget, None, idx))
            text_box.connect("focus-out-event", lambda widget, event, idx=i: self.on_text_box_updated(widget, event, idx))
            vbox4.pack_start(text_box, True, True, 0)
            self.text_boxes.append(text_box)

            # Create and add a progress bar with vertical alignment
            progress_bar = Gtk.ProgressBar()
            progress_bar.set_fraction(0.0)  # Set initial value, adjust as needed
            progress_bar.set_valign(Gtk.Align.CENTER)  # Center align vertically
            vbox5.pack_start(progress_bar, True, True, 0)
            self.probability_progress_bars.append(progress_bar)

        # Create a horizontal box to hold the vertical boxes
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        hbox.pack_start(vbox1, False, False, 0)
        hbox.pack_start(vbox2, False, False, 0)
        hbox.pack_start(vbox3, False, False, 0)
        hbox.pack_start(vbox4, True, True, 0)
        hbox.pack_start(vbox5, True, True, 0)

        # Schedule the update_progress_bars method to be called every half a second
        GLib.timeout_add(500, self.update_progress_bars)

        # Add the horizontal box to the main vertical box
        self.ui_vbox.pack_start(hbox, False, False, 0)

    def update_text_boxes(self):
        """Sets the text boxes to align wit updated entries."""
        for i, entry in enumerate(self.text_image_matcher.entries):
            self.text_boxes[i].set_text(entry.text)
            self.negative_check_buttons[i].set_active(entry.negative)
            self.ensemble_check_buttons[i].set_active(entry.ensemble)
    
    def update_text_prefix(self, new_text_prefix):
        """Updates the text_prefix labels in the UI."""
        self.text_image_matcher.text_prefix = new_text_prefix  # Update the instance variable
        for label in self.text_prefix_labels:
            label.set_text(new_text_prefix)

    # UI Callbacks
    def quit_button_clicked(self, widget):
        print("Quit button clicked")
        self.shutdown()

    def on_text_box_updated(self, widget, event, idx):
        """Callback function for text box updates."""
        text = widget.get_text()
        print(f"Text box {idx} updated: {text}")
        self.text_image_matcher.add_text(widget.get_text(), idx)

    def on_slider_value_changed(self, widget):
        value = float(widget.get_value())
        print(f"Setting detection threshold to: {value}")
        self.text_image_matcher.set_threshold(value)

    def on_negative_check_button_toggled(self, widget, idx):
        negative = widget.get_active()
        print(f"Text box {idx} is set to negative: {negative}")
        self.text_image_matcher.entries[idx].negative = negative

    def on_ensemble_check_button_toggled(self, widget, idx):
        ensemble = widget.get_active()
        print(f"Text box {idx} is set to ensemble: {ensemble}")
        # Encode text with new ensemble option
        self.text_image_matcher.add_text(self.text_boxes[idx].get_text(), idx, ensemble=ensemble)

    def on_load_button_clicked(self, widget):
        """Callback function for the load button."""
        print(f"Loading embeddings from {self.json_file}\n")
        self.text_image_matcher.load_embeddings(self.json_file)
        self.update_text_boxes()
        self.slider.set_value(self.text_image_matcher.threshold)
        self.update_text_prefix(self.text_image_matcher.text_prefix)

    def on_save_button_clicked(self, widget):
        """Callback function for the save button."""
        print(f"Saving embeddings to {self.json_file}\n")
        self.text_image_matcher.save_embeddings(self.json_file)

    def update_progress_bars(self):
        """Updates the progress bars based on the current probability values."""
        for i, entry in enumerate(self.text_image_matcher.entries):
            if entry.text != "":
                self.probability_progress_bars[i].set_fraction(entry.probability)
            else:
                self.probability_progress_bars[i].set_fraction(0.0)
        return True

    def add_default_texts(self):
        if (self.multi_stream):
            self.text_image_matcher.add_text("man",0, True)
            self.text_image_matcher.add_text("woman",1, True)
            self.text_image_matcher.add_text("man with striped shirt",2)
        else:
            if (self.detector == "person"):
                self.text_image_matcher.add_text("person",0, True) # Default entry for object detection (background)
                self.text_image_matcher.add_text("person with a water bottle",1)
                self.text_image_matcher.add_text("person with a hat ",2)
                self.text_image_matcher.add_text("man with a bag",3)
                self.text_image_matcher.add_text("woman with glasses",4)
            elif (self.detector == "face"):
                self.text_image_matcher.add_text("face",0, True) # Default entry for object detection (background)
                self.text_image_matcher.add_text("smiling face",1)
                self.text_image_matcher.add_text("person rising eye brows",2)
                self.text_image_matcher.add_text("person winking his eye",3)
            elif (self.detector == "fast_sam"):
                self.text_image_matcher.add_text("object",0,True) # Default entry for object detection (background)
                self.text_image_matcher.add_text("cell phone",1)
                self.text_image_matcher.add_text("person",2)
                self.text_image_matcher.add_text("car",3)
                self.text_image_matcher.add_text("table",4)
                self.text_image_matcher.add_text("computer",5)
            elif (self.detector == "none"):
                self.text_image_matcher.add_text("empty room",0, True) # Default entry for object detection (background
                self.text_image_matcher.add_text("person",1)
                self.text_image_matcher.add_text("cellphone",2)
    
    def update_video_muxing(self):
            selected_stream = self.text_image_matcher.user_data
            if (selected_stream is not None):
                # pasre stream id to get stream number example is SRC_0
                stream_number = int(selected_stream.split("_")[1])
                self.input_selector_element.set_property('active-pad', self.input_selector_element.get_static_pad(f'sink_{stream_number}'))
            return True
        
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
        if self.multi_stream:
            pipeline_str = get_pipeline_multi(self.current_path, self.detector, self.sync, self.input_uri, self.tappas_workspace, self.tappas_version)
        else:
            pipeline_str = get_pipeline(self.current_path, self.detector, self.sync, self.input_uri, self.tappas_workspace, self.tappas_version)
        print(f'PIPELINE:\ngst-launch-1.0 {pipeline_str}')
        # run parse_launch and handle errors
        try:
            pipeline = Gst.parse_launch(pipeline_str)
        except GLib.Error as e:
            logger.error(f"An error occurred while parsing the pipeline: {e})")
        return pipeline
    
if __name__ == "__main__":
    main()
