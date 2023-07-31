#include "example_app_utils.hpp"

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
      // convert the temperature and power to strings with 2 decimal places
      std::stringstream power_stream;
      power_stream << std::fixed << std::setprecision(2) << power_float;
      std::stringstream temperature_stream;
      temperature_stream << std::fixed << std::setprecision(2) << temperature_float;
      std::string text = "Temperature: " + temperature_stream.str() + " Power: " + power_stream.str();
      std::cout << text << std::endl;
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
  char result[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
  if (count != -1)
  {
    // remove the executable name
    for (int i = count; i >= 0; i--)
    {
      if (result[i] == '/')
      {
        result[i + 1] = '\0';
        break;
      }
    }
  }
  return std::string(result);
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
void setup_hailo_utils(GstElement *pipeline, GstBus *bus, GMainLoop *main_loop, UserData *user_data, cxxopts::ParseResult result)
{

  // Set the pipeline element
  user_data->pipeline = pipeline;

  // Set the main loop element
  user_data->main_loop = main_loop;

  // Extract closing messages
  gst_bus_add_watch(bus, async_bus_callback, user_data);
  return;
}
