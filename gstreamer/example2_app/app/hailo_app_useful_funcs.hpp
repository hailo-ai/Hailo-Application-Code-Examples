/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
* Helper functions based on https://github.com/agrechnev/gst_app_tutorial
**/


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
    static GstClockTime last_time = 0;
    static int framecount = 0;
    framecount++;
    GstClockTime current_time = gst_clock_get_time(gst_system_clock_obtain());
    GstClockTimeDiff diff = current_time - last_time;
    if (diff >= GST_SECOND) {
        std::cout << "Framerate: " << framecount / GST_TIME_AS_SECONDS(diff) << std::endl;
        framecount = 0;
        last_time = current_time;
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

//======================================================================================================================
// A few useful routines for diagnostics

static gboolean printField(GQuark field, const GValue *value, gpointer pfx) {
    using namespace std;
    gchar *str = gst_value_serialize(value);
    cout << (char *) pfx << " " << g_quark_to_string(field) << " " << str << endl;
    g_free(str);
    return TRUE;
}

void printCaps(const GstCaps *caps, const std::string &pfx) {
    using namespace std;
    if (caps == nullptr)
        return;
    if (gst_caps_is_any(caps))
        cout << pfx << "ANY" << endl;
    else if (gst_caps_is_empty(caps))
        cout << pfx << "EMPTY" << endl;
    for (int i = 0; i < gst_caps_get_size(caps); ++i) {
        GstStructure *s = gst_caps_get_structure(caps, i);
        cout << pfx << gst_structure_get_name(s) << endl;
        gst_structure_foreach(s, &printField, (gpointer) pfx.c_str());
    }
}


void printPadsCB(const GValue * item, gpointer userData) {
    using namespace std;
    GstElement *element = (GstElement *)userData;
    GstPad *pad = (GstPad *)g_value_get_object(item);
    myAssert(pad);
    cout << "PAD : " << gst_pad_get_name(pad) << endl;
    GstCaps *caps = gst_pad_get_current_caps(pad);
    
    char * str = gst_caps_to_string(caps);
    cout << str << endl;
    free(str);
}

void printPads(GstElement *element) {
    using namespace std;
    GstIterator *pad_iter = gst_element_iterate_pads(element);
    //delay to allow caps negotiation to complete
    g_usleep(100000);
    gst_iterator_foreach(pad_iter, printPadsCB, element);
    gst_iterator_free(pad_iter);

}
void diagnose(GstElement *element) {
    using namespace std;
    cout << "=====================================" << endl;
    cout << "DIAGNOSE element : " << gst_element_get_name(element) << endl;
    printPads(element);
    cout << "=====================================" << endl;
}

void parse_state_change_message(GstMessage *msg) {
    GstState sOld, sNew, sPenging;
    gst_message_parse_state_changed(msg, &sOld, &sNew, &sPenging);
    std::string elem_name = GST_OBJECT_NAME(GST_MESSAGE_SRC(msg));
    std::cout << "Element " << elem_name << " changed from " << 
    gst_element_state_get_name(sOld) << " to " <<
    gst_element_state_get_name(sNew) << std::endl;
}

#endif // HAILO_APP_USEFUL_FUNCS_HPP