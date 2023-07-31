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
};

//******************************************************************
// BUS CALLBACK function
//******************************************************************

gboolean async_bus_callback(GstBus *bus, GstMessage *message, gpointer user_data);

//******************************************************************
// MAIN utility functions
//******************************************************************
cxxopts::Options build_arg_parser();
std::string getexepath();

void setup_hailo_utils(GstElement *pipeline, GstBus *bus, GMainLoop *main_loop, UserData *user_data, cxxopts::ParseResult result);
