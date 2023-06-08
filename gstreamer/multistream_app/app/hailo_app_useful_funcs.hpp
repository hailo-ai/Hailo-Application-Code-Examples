#include <iostream>
#include <string>
#include <gst/gst.h>
#include <cxxopts.hpp>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <glib.h>


// Tappas includes
#include "hailo_objects.hpp"
#include "hailo_common.hpp"
#include "gst_hailo_meta.hpp"

#include "SrcBin.hpp"
//******************************************************************
// DATA TYPES
//******************************************************************

struct UserData
{
  GstElement *pipeline;
  GMainLoop *main_loop;
  // a vector of src bins
  std::vector<SrcBin*> src_bins;
};
//******************************************************************
// BUS CALLBACK function
//******************************************************************

gboolean async_bus_callback(GstBus *bus, GstMessage *message, gpointer user_data);

//******************************************************************
// PROBE CALLBACK examples
//******************************************************************

/*
These examples can be connected to identity elements in the pipeline connecting to "handoff" signal
here is an example of how to connect a probe to an identity element in the pipeline
// get fps_probe element
GstElement *fps_probe = gst_bin_get_by_name(GST_BIN(pipeline), "fps_probe");
// set probe callback
g_signal_connect(fps_probe, "handoff", G_CALLBACK(fps_probe_callback), &user_data);
*/
void fps_probe_callback(GstElement *element, GstBuffer *buffer, gpointer user_data);
void pts_probe_callback(GstElement *element, GstBuffer *buffer, gpointer user_data);
void probe_callback(GstElement *element, GstBuffer *buffer, gpointer user_data);
GstPadProbeReturn events_debug_pad_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);

//******************************************************************
// UTILITY FUNCTIONS
//******************************************************************
void set_queue_properties(GstElement *queue, gboolean leaky, guint max_size_buffers);
void disable_qos_in_bin(GstBin *bin);

gboolean check_pipeline_state(GstElement* pipeline, GstState target_state, GstClockTime timeout);

//******************************************************************
// MAIN utility functions
//******************************************************************
cxxopts::Options build_arg_parser();
std::string getexepath();
