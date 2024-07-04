import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GLib

def build_ui(self, args):
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
    # Add label and track_id input box
    self.track_id_label = Gtk.Label(label="Track ID")
    self.track_id_entry = Gtk.Entry()
    hbox.pack_end(self.track_id_entry, False, False, 0)
    hbox.pack_end(self.track_id_label, False, False, 0)
    self.track_id_entry.connect("activate", lambda widget: self.on_track_id_update(widget))
    self.track_id_entry.connect("focus-out-event", lambda widget, event: self.on_track_id_update(widget))

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
        progress_bar.set_fraction(0.0)
        progress_bar.set_valign(Gtk.Align.CENTER)
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

    self.ui_vbox.pack_start(hbox, False, False, 0)


def update_text_boxes(self):
    for i, entry in enumerate(self.text_image_matcher.entries):
        self.text_boxes[i].set_text(entry.text)
        self.negative_check_buttons[i].set_active(entry.negative)
        self.ensemble_check_buttons[i].set_active(entry.ensemble)


def update_text_prefix(self, new_text_prefix):
    self.text_image_matcher.text_prefix = new_text_prefix
    for label in self.text_prefix_labels:
        label.set_text(new_text_prefix)


def quit_button_clicked(self, widget):
    print("Quit button clicked")
    self.shutdown()


def on_text_box_updated(self, widget, event, idx):
    """Callback function for text box updates."""
    text = widget.get_text()
    print(f"Text box {idx} updated: {text}")
    self.text_image_matcher.add_text(widget.get_text(), idx)

def on_track_id_update(self, widget):
    """Callback function for track id updates."""
    track_id_focus = widget.get_text()
    # check if track id is a number
    if not track_id_focus.isdigit():
        print(f"Track ID must be a number, got: {track_id_focus}")
        widget.set_text("")
        self.text_image_matcher.track_id_focus = None
        return
    print(f"Track ID updated: {track_id_focus}")
    self.text_image_matcher.track_id_focus = int(track_id_focus)

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
            self.probability_progress_bars[i].set_fraction(entry.tracked_probability)
        else:
            self.probability_progress_bars[i].set_fraction(0.0)
    return True

