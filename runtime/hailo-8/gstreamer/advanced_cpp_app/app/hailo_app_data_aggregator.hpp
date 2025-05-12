#ifndef HAILO_APP_DATA_AGGREGATOR_HPP
#define HAILO_APP_DATA_AGGREGATOR_HPP

// General cpp includes
#include <chrono>
#include <condition_variable>
#include <cxxopts.hpp>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/app/gstappsink.h>
#include <iostream>
#include <mutex>
#include <chrono>
#include <ctime>
#include <shared_mutex>
#include <stdio.h>
#include <thread>
#include <unistd.h>
#include <glib.h>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <sys/resource.h>
#include <limits.h>
#include <iomanip>

// Tappas includes
#include "hailo_objects.hpp"
#include "hailo_common.hpp"

//******************************************************************
// DATA TYPES
//******************************************************************

struct AggData {
  GstElement* pipeline;
  GMainLoop* main_loop;
  GstElement* text_overlay;
  gboolean print_fps;
  gboolean print_hailo_stats;
  gboolean print_host_stats;
  gboolean print_required;
  gboolean print_to_terminal;
  pid_t pid;
};


//******************************************************************
// DATA Aggregator
//******************************************************************

class DataAggregator {
public:
  // constructor
  DataAggregator(){}; // Default constructor
  DataAggregator(gpointer agg_data);

  void initilize_agg_data(gpointer agg_data);
  void set_power(double power);

  void set_fps(double fps);

  void set_temp(double temp);

  void set_cpu(double cpu);

  void set_mem(double mem);

  std::string get_string();

  void display_string();
  
  pid_t get_pid();

private:
  void update_string();

  std::mutex mutex_;
  double power_ = 0.0;
  double fps_ = 0.0;
  double temp_ = 0.0;
  double cpu_ = 0.0;
  double mem_ = 0.0;
  std::string data_string_;
  AggData *data_;
};

//******************************************************************
// PIPELINE UTILITIES
//******************************************************************
/**
 * @brief callback of new fps measurement signal
 *
 * @param fpsdisplaysink the element who sent the signal
 * @param fps the fps measured
 * @param droprate drop rate measured
 * @param avgfps average fps measured
 * @param udata extra data from the user
 */
static void fps_measurements_callback(GstElement *fpsdisplaysink,
                                      gdouble fps,
                                      gdouble droprate,
                                      gdouble avgfps,
                                      gpointer udata);

double getProcessCpuUsage(int pid);

double getProcessMemoryUsage(int pid);

bool display_stats_callback(gpointer udata);

bool update_host_stats_callback(gpointer udata);

void add_aggregator_options(cxxopts::Options &options);

/**
 * @brief Setup the Hailo utils.
 *        This function shoud be called from main to setup Bus Watch and User Data
 * @param pipeline 
 *        The GStreamer pipeline.
 * @param bus 
 *        The GStreamer bus.
 * @param main_loop 
 *        The GStreamer main loop.
 * @param print_fps 
 *        Enable displaying FPS.
 * @param print_hailo_stats 
 *        Enable displaying Hailo stats.
 * @param print_host_stats 
 *        Enable displaying host stats.
 * @param print_to_terminal 
 *        When set will print to terminal not overlay video.
 * @return AggData 
 *         The user data.
 */
// void setup_hailo_data_aggregator(gpointer user_data, cxxopts::ParseResult result)
DataAggregator* setup_hailo_data_aggregator(GstElement *pipeline, GMainLoop *main_loop, cxxopts::ParseResult result);

#endif // HAILO_APP_DATA_AGGREGATOR_HPP