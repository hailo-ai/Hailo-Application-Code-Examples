#include "hailo_app_useful_funcs.hpp"
#include "hailo_app_data_aggregator.hpp"

//******************************************************************
// BUS CALLBACK function
//******************************************************************

gboolean async_bus_callback(GstBus *bus, GstMessage *message, gpointer user_data)
{
  UserData *data = static_cast<UserData *>(user_data);
  switch (GST_MESSAGE_TYPE(message))
  {
  case GST_MESSAGE_ERROR:
  {
    // An error occurred in the pipeline
    GError *error = nullptr;
    gchar *debug_info = nullptr;
    gst_message_parse_error(message, &error, &debug_info);
    g_printerr("Error received from element %s: %s\n", GST_OBJECT_NAME(message->src), error->message);
    g_printerr("Debugging info: %s\n", debug_info ? debug_info : "none");
    g_clear_error(&error);
    g_free(debug_info);
    // stop main loop
    g_main_loop_quit(data->main_loop);
    break;
  }
  case GST_MESSAGE_EOS:
    // The pipeline has reached the end of the stream
    g_print("End-Of-Stream reached.\n");
    // stop main loop
    g_main_loop_quit(data->main_loop);
    break;
  case GST_MESSAGE_ELEMENT:
  {
    const GstStructure *structure = gst_message_get_structure(message);
    // if structure name is HailoDeviceStatsMessage
    if (gst_structure_has_name(structure, "HailoDeviceStatsMessage"))
    {
      // get the temperature
      const GValue *temperature = gst_structure_get_value(structure, "temperature");
      // get the temperature as a float
      gfloat temperature_float = g_value_get_float(temperature);
      // get the power
      const GValue *power = gst_structure_get_value(structure, "power");
      // get the power as a float
      gfloat power_float = g_value_get_float(power);
      if (data->data_aggregator != nullptr)
      {
        DataAggregator *data_aggregator = static_cast<DataAggregator *>(data->data_aggregator);
        data_aggregator->set_temp(temperature_float);
        data_aggregator->set_power(power_float);
      }
      else
      {
        // convert the temperature and power to strings with 2 decimal places
        std::stringstream power_stream;
        power_stream << std::fixed << std::setprecision(2) << power_float;
        std::stringstream temperature_stream;
        temperature_stream << std::fixed << std::setprecision(2) << temperature_float;
        std::string text = "Temperature: " + temperature_stream.str() + " Power: " + power_stream.str();
        std::cout << text << std::endl;
      }
    }
    break;
  }
  // print QOS message
  case GST_MESSAGE_QOS:
  {
    std::cout << "QOS message detected from " << GST_OBJECT_NAME(message->src) << std::endl;
    break;
  }
  default:
    // Print a message for other message types
    // g_print("Received message of type %s\n", GST_MESSAGE_TYPE_NAME(message));
    break;
  }
  // We want to keep receiving messages
  return TRUE;
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
  
  if (diff >= GST_SECOND)
  {
    g_print("Framerate %s: %d\n", element_name.c_str(), framecount_map[element_name] / GST_TIME_AS_SECONDS(diff));
    framecount_map[element_name] = 0;
    last_time_map[element_name] = current_time;
  }
}

void detections_probe_callback(GstElement *element, GstBuffer *buffer, gpointer user_data)
{
    HailoROIPtr roi;
    std::vector<HailoDetectionPtr> detections;
    roi = get_hailo_main_roi(buffer, true);
    detections = hailo_common::get_hailo_detections(roi);
    
    //print detections
    for (auto detection : detections)
    {
        std::cout << "Probe Detection: " << detection->get_label() << " " << detection->get_confidence() << std::endl; 
    }
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
void disable_qos_in_bin(GstBin *bin)
{
  GstIterator *it;
  GValue item = G_VALUE_INIT;
  gboolean done = FALSE;

  g_return_if_fail(GST_IS_BIN(bin));

  it = gst_bin_iterate_elements(bin);
  while (!done)
  {
    switch (gst_iterator_next(it, &item))
    {
    case GST_ITERATOR_OK:
    {
      GstElement *element = GST_ELEMENT(g_value_get_object(&item));

      if (g_object_class_find_property(G_OBJECT_GET_CLASS(element), "qos"))
      {
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


/**
 * Function: check_pipeline_state
 * ------------------------------
 * This function checks if all elements in a GStreamer pipeline have reached a target state within a specified timeout period.
 *
 * Parameters:
 * - pipeline: A pointer to a GstElement representing the GStreamer pipeline.
 * - target_state: The GstState that all elements in the pipeline should have reached.
 * - timeout: The maximum amount of time to wait for the state change, specified in nanoseconds.
 *
 * Returns:
 * - A boolean value indicating whether all elements in the pipeline have reached the target state within the timeout period.
 *   Returns TRUE if all elements are in the target state, FALSE otherwise.
 *
 * The function works by iterating over all elements in the pipeline and checking their state individually.
 * If it encounters an element that hasn't reached the target state within the timeout period, it returns FALSE immediately.
 * If all elements are in the target state, or if the pipeline doesn't contain any elements, the function returns TRUE.
 *
 * Note:
 * The timeout is applied to each element individually, so the total amount of time the function can block is the timeout multiplied
 * by the number of elements in the pipeline.
 */

gboolean check_pipeline_state(GstElement* pipeline, GstState target_state, GstClockTime timeout) {
    GstStateChangeReturn ret;
    GstIterator *it = gst_bin_iterate_elements(GST_BIN(pipeline));
    GValue item = G_VALUE_INIT;
    gboolean done = FALSE, result = TRUE;
    GstElement *element;

    while (!done) {
        switch (gst_iterator_next(it, &item)) {
        case GST_ITERATOR_OK:
            element = GST_ELEMENT(g_value_get_object(&item));
            GstState state;
            ret = gst_element_get_state(element, &state, NULL, timeout);
            if (ret != GST_STATE_CHANGE_SUCCESS || state != target_state) {
                g_warning("Element %s is not in %s state", GST_ELEMENT_NAME(element), gst_element_state_get_name(target_state));
                result = FALSE;
                done = TRUE; // Stop checking as soon as we find one element that hasn't reached the target state
            }
            g_value_reset(&item);
            break;
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

    return result;
}

//******************************************************************
// MAIN utility functions
//******************************************************************
/**
 * @brief Build command line arguments.
 * 
 * @return cxxopts::Options 
 *         The available user arguments.
 */
cxxopts::Options build_arg_parser()
{
  cxxopts::Options options("Hailo App");
  options.allow_unrecognised_options();
  options.add_options()
  ("h, help", "Show this help")
  ("i, input", "Set the input source.\nSource can be a device for example /dev/video0 \nOr URI for file use file://<FILE-FULL_PATH>\nSee README for online URI source example\n", cxxopts::value<std::string>()->default_value(std::string("/dev/video0")))
  ("s, hailo-stats", "Enable displaying Hailo stats", cxxopts::value<bool>()->default_value("false"))
  ("sync-pipeline", "Enable to sync to video framerate otherwise runs as fast as possible", cxxopts::value<bool>()->default_value("false"))
  ("f, show-fps", "Enable displaying FPS", cxxopts::value<bool>()->default_value("false"));
  return options;
}

/**
 * @brief Get the executable path.
 * 
 * @return std::string
*/
std::string getexepath()
{
    char result[ PATH_MAX ];
    ssize_t count = readlink( "/proc/self/exe", result, PATH_MAX );
    if (count != -1) {
        // remove the executable name
        for (int i = count; i >= 0; i--) {
            if (result[i] == '/') {
                result[i+1] = '\0';
                break;
            }
        }
    }
    return std::string( result );
}

/**
 * @brief Setup the Hailo utils.
 *        This function shoud be called from main to setup Bus Watch and User Data
 * @param pipeline 
 *        The GStreamer pipeline.
 * @param bus 
 *        The GStreamer bus.
 * @param main_loop 
 *        The GStreamer main loop.
 */
void setup_hailo_utils(GstElement *pipeline, GstBus *bus, GMainLoop *main_loop, UserData* user_data, cxxopts::ParseResult result)
{
  // Set the pipeline element
  user_data->pipeline = pipeline;

  // Set the main loop element
  user_data->main_loop = main_loop;
  
  // Extract closing messages
  gst_bus_add_watch(bus, async_bus_callback, user_data);
  return;
}
