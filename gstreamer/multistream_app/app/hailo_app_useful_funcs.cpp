#include "hailo_app_useful_funcs.hpp"

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
void fps_probe_callback(GstElement *element, GstBuffer *buffer, gpointer user_data)
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
        g_print("Framerate %s: %d\n", element_name.c_str(), framecount_map[element_name] / GST_TIME_AS_SECONDS(diff));
        framecount_map[element_name] = 0;
        last_time_map[element_name] = current_time;
    }
    //print buffer PTS
    GST_INFO("Buffer PTS: %s: %" GST_TIME_FORMAT, element_name.c_str(), GST_TIME_ARGS(GST_BUFFER_PTS(buffer)));
} 

void probe_callback(GstElement *element, GstBuffer *buffer, gpointer user_data)
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

// Probe callback debug events
GstPadProbeReturn
events_debug_pad_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  GstEvent *event = GST_PAD_PROBE_INFO_EVENT(info);
  const gchar *event_name = GST_EVENT_TYPE_NAME(event);
  GstPadDirection direction = gst_pad_get_direction(pad);
  const gchar *direction_name = (direction == GST_PAD_SRC) ? "src" : "sink";
  GstElement *parent = gst_pad_get_parent_element(pad);

  GST_DEBUG("Event '%s' on %s pad '%s', parent element '%s'\n",
    event_name,
    direction_name,
    GST_PAD_NAME(pad),
    GST_ELEMENT_NAME(parent));

  gst_object_unref(parent);

  return GST_PAD_PROBE_OK;
}

//******************************************************************
// UTILITY FUNCTIONS
//******************************************************************

// a function to set queue properties
void set_queue_properties(GstElement *queue, gboolean leaky = false, guint max_size_buffers = 3)
{
    g_object_set(G_OBJECT(queue), "leaky", leaky, NULL);
    g_object_set(G_OBJECT(queue), "max-size-buffers", max_size_buffers, NULL);
    g_object_set(G_OBJECT(queue), "max-size-bytes", 0, NULL);
    g_object_set(G_OBJECT(queue), "max-size-time", 0, NULL);
}
/**
 * This function is used to disable the Quality of Service (QoS) events on all elements of a GStreamer pipeline (or bin).
 * QoS events in GStreamer provide a mechanism for elements to provide feedback about the quality of the stream. 
 * However, in certain cases, we might want to disable this feature for performance or for handling specific use cases. 
 *
 * @param bin Pointer to the GstBin object, which is a container element that handles GStreamer elements as its child objects. 
 *            This GstBin will be searched recursively and QoS will be disabled on all found elements that have QoS capability.
 */
void disable_qos_in_bin(GstBin *bin) {
    GstIterator *it;
    GValue item = G_VALUE_INIT;
    gboolean done = FALSE;

    g_return_if_fail(GST_IS_BIN(bin));

    it = gst_bin_iterate_elements(bin);
    while (!done) {
        switch (gst_iterator_next(it, &item)) {
        case GST_ITERATOR_OK: {
            GstElement *element = GST_ELEMENT(g_value_get_object(&item));

            if (g_object_class_find_property(G_OBJECT_GET_CLASS(element), "qos")) {
                g_object_set(G_OBJECT(element), "qos", FALSE, NULL);
            }

            g_value_reset(&item);
            break;
        }

        case GST_ITERATOR_RESYNC:
            gst_iterator_resync(it);
            break;

        case GST_ITERATOR_ERROR:
        case GST_ITERATOR_DONE:
            done = TRUE;
            break;
        }
    }
    g_value_unset(&item);
    gst_iterator_free(it);
}

