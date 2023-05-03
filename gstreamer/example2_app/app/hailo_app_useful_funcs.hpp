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

#endif // HAILO_APP_USEFUL_FUNCS_HPP
