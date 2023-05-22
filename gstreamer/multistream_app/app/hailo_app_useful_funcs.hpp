#ifndef HAILO_APP_USEFUL_FUNCS_HPP
#define HAILO_APP_USEFUL_FUNCS_HPP

#include <iostream>
#include <string>
#include <gst/gst.h>

// Tappas includes
#include "hailo_objects.hpp"
#include "hailo_common.hpp"
#include "gst_hailo_meta.hpp"

// A simple assertion function
inline void myAssert(bool b, const std::string &s = "MYASSERT ERROR !") {
    if (!b)
        throw std::runtime_error(s);
}

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
static void fps_probe_callback(GstElement *element, GstBuffer *buffer, gpointer user_data)
{
    // use maps for these static variables to support multiple streams
    static std::map<std::string, GstClockTime> last_time_map;
    static std::map<std::string, int> framecount_map;
    // get element name
    std::string element_name = gst_element_get_name(element);
    // check if element name is in map
    if (last_time_map.find(element_name) == last_time_map.end())
    {
        // if not in map, add it
        last_time_map[element_name] = 0;
        framecount_map[element_name] = 0;
    }
    
    framecount_map[element_name]++;
    GstClockTime current_time = gst_clock_get_time(gst_system_clock_obtain());
    GstClockTimeDiff diff = current_time - last_time_map[element_name];
    if (diff >= GST_SECOND) {
        std::cout << "Framerate " << element_name << ": " << framecount_map[element_name] / GST_TIME_AS_SECONDS(diff) << std::endl;
        framecount_map[element_name] = 0;
        last_time_map[element_name] = current_time;
    }
    //print buffer PTS
    std::cout << "Buffer PTS: " << element_name << ": "<< GST_BUFFER_PTS(buffer) << std::endl;
} 

static void probe_callback(GstElement *element, GstBuffer *buffer, gpointer user_data)
{
    HailoROIPtr roi;
    std::vector<HailoDetectionPtr> detections;
    std::vector<HailoDetectionPtr> sub_detections;
    roi = get_hailo_main_roi(buffer, true);
    detections = hailo_common::get_hailo_detections(roi);
    
    //print detections
    for (auto detection : detections)
    {
        std::cout << "Probe Detection: " << detection->get_label() << " " << detection->get_confidence() << std::endl; 
    }
}

void handle_pipeline_failure(GstBin* pipeline) {
 
    // Get the error message
    const gchar* errmsg = gst_element_state_change_return_get_name(GST_STATE_CHANGE_FAILURE);
    g_warning("Failed to set pipeline state to playing: %s", errmsg);

    // Check the state of each element in the pipeline
    GstState state, pending;
    GstIterator *iter = gst_bin_iterate_elements(GST_BIN(pipeline));
    GValue elem = G_VALUE_INIT;
    while (gst_iterator_next(iter, &elem) == GST_ITERATOR_OK)
    {
      GstElement *element = GST_ELEMENT(g_value_get_object(&elem));
      gst_element_get_state(element, &state, &pending, 0);
      g_message("Element %s state: %s", GST_ELEMENT_NAME(element), gst_element_state_get_name(state));
      gst_element_set_state(element, GST_STATE_NULL);
      gst_object_unref(element);
    }
    g_value_unset(&elem);
    gst_iterator_free(iter);

    // Generate a DOT graph of the pipeline for debugging
    //   GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "pipeline");
    gst_debug_bin_to_dot_file(GST_BIN(pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "pipeline_failed");
    // Set up a pipeline bus and get error messages
    GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    GstMessage *msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE, GST_MESSAGE_ERROR);
    if (msg != NULL)
    {
      GError *err = NULL;
      gchar *debug_info = NULL;
      gst_message_parse_error(msg, &err, &debug_info);
      g_warning("Pipeline error: %s", err->message);
      g_error_free(err);
      g_free(debug_info);
      gst_message_unref(msg);
    }
    gst_object_unref(bus);
  }

#endif // HAILO_APP_USEFUL_FUNCS_HPP
