#ifndef HAILO_APP_USEFUL_FUNCS_HPP
#define HAILO_APP_USEFUL_FUNCS_HPP

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

//******************************************************************
// DATA TYPES
//******************************************************************

struct UserData
{
  GstElement *pipeline;
  GMainLoop *main_loop;
  void* data_aggregator = nullptr;
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
void detections_probe_callback(GstElement *element, GstBuffer *buffer, gpointer user_data);

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

void setup_hailo_utils(GstElement *pipeline, GstBus *bus, GMainLoop *main_loop, UserData *user_data, cxxopts::ParseResult result);

#endif // HAILO_APP_USEFUL_FUNCS_HPP