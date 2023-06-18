//******************************************************************
// DATA Aggregator
//******************************************************************
#include "hailo_app_data_aggregator.hpp"

DataAggregator::DataAggregator(gpointer agg_data)
{
  this->initilize_agg_data(agg_data);
}

void DataAggregator::initilize_agg_data(gpointer agg_data)
{
  std::cout << "Initializing aggregator data" << std::endl;
  this->data_ = static_cast<AggData *>(agg_data);
}

void DataAggregator::set_power(double power)
{
  std::lock_guard<std::mutex> lock(mutex_);
  power_ = power;
  update_string();
}

void DataAggregator::set_fps(double fps)
{
  std::lock_guard<std::mutex> lock(mutex_);
  fps_ = fps;
  update_string();
}

void DataAggregator::set_temp(double temp)
{
  std::lock_guard<std::mutex> lock(mutex_);
  temp_ = temp;
  update_string();
}

void DataAggregator::set_cpu(double cpu)
{
  std::lock_guard<std::mutex> lock(mutex_);
  cpu_ = cpu;
  update_string();
}

void DataAggregator::set_mem(double mem)
{
  std::lock_guard<std::mutex> lock(mutex_);
  mem_ = mem;
  update_string();
}

std::string DataAggregator::get_string()
{
  std::lock_guard<std::mutex> lock(mutex_);
  return data_string_;
}
void DataAggregator::display_string()
{
  if (data_->print_to_terminal)
  {
    std::cout << get_string().c_str() << std::endl;
  }
  else
  {
    // set the text on the display
    g_object_set(G_OBJECT(data_->text_overlay), "text", get_string().c_str(), NULL);
  }
}

pid_t DataAggregator::get_pid()
{
  return data_->pid;
}

// private:
void DataAggregator::update_string()
{
  std::stringstream ss;
  ss << std::fixed << std::setprecision(2);
  if (data_->print_fps)
  {
    ss << "FPS: " << fps_ << " ";
  }
  if (data_->print_hailo_stats)
  {
    ss << "Power: " << power_ << "W, Temp: " << temp_ << "C ";
  }
  if (data_->print_host_stats)
  {
    ss << "CPU: " << cpu_ << "%, MEM: " << mem_ << "MB ";
  }
  data_string_ = ss.str();
}

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
                                      gpointer udata)
{
  DataAggregator *data_aggregator = static_cast<DataAggregator *>(udata);
  data_aggregator->set_fps(fps);
}

double getProcessCpuUsage(int pid)
{
  static double prev_cpu_usage = 0.0;     // Static variable to store previous measurement
  static double prev_sys_cpu_usage = 0.0; // Static variable to store previous system CPU usage

  // Open the stat file for the process
  std::ifstream procStat("/proc/" + std::to_string(pid) + "/stat");
  std::string line;
  std::getline(procStat, line);

  // Split the line into fields
  std::vector<std::string> fields;
  std::stringstream lineStream(line);
  std::string field;
  while (std::getline(lineStream, field, ' '))
  {
    fields.push_back(field);
  }

  // Parse the fields to get the process's CPU usage information
  double user_time = std::stod(fields[13]);
  double system_time = std::stod(fields[14]);
  double child_user_time = std::stod(fields[15]);
  double child_system_time = std::stod(fields[16]);

  // Close the stat file for the process
  procStat.close();

  // Open the stat file for the system
  procStat.open("/proc/stat");
  std::getline(procStat, line);

  // Split the line into fields
  fields.clear();
  lineStream.str(line);
  lineStream.clear();
  while (std::getline(lineStream, field, ' '))
  {
    fields.push_back(field);
  }

  // Parse the fields to get the system's CPU usage information
  double sys_user_time = std::stod(fields[2]);
  double sys_nice_time = std::stod(fields[3]);
  double sys_system_time = std::stod(fields[4]);
  double sys_idle_time = std::stod(fields[5]);

  // Close the stat file for the system
  procStat.close();

  // Calculate the change in CPU usage
  double cpu_usage = (user_time + system_time + child_user_time + child_system_time) - prev_cpu_usage;
  double sys_cpu_usage = (sys_user_time + sys_nice_time + sys_system_time + sys_idle_time) - prev_sys_cpu_usage;
  double cpu_usage_percent = (cpu_usage / sys_cpu_usage) * 100.0;

  // Update the static variables for the next measurement
  prev_cpu_usage = user_time + system_time + child_user_time + child_system_time;
  prev_sys_cpu_usage = sys_user_time + sys_nice_time + sys_system_time + sys_idle_time;

  return cpu_usage_percent;
}

double getProcessMemoryUsage(int pid)
{
  ///////////////////////////////////////////////////////////////////////////////
  // Note that the memory usage returned by this function is the maximum memory used
  ///////////////////////////////////////////////////////////////////////////////

  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  // Convert the memory usage from KB to MB
  return usage.ru_maxrss / 1024.0;
}

bool display_stats_callback(gpointer udata)
{
  DataAggregator *data_aggregator = static_cast<DataAggregator *>(udata);
  data_aggregator->display_string();
  return true;
}

bool update_host_stats_callback(gpointer udata)
{
  DataAggregator *data_aggregator = static_cast<DataAggregator *>(udata);
  try
  {
    // Get CPU usage
    double cpu_usage = getProcessCpuUsage(data_aggregator->get_pid());
    data_aggregator->set_cpu(cpu_usage);
    // Get memory usage
    double memory_usage = getProcessMemoryUsage(data_aggregator->get_pid());
    data_aggregator->set_mem(memory_usage);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error on update_host_stats_callback: " << e.what() << std::endl;
    return false;
  }
  return true;
}

void add_aggregator_options(cxxopts::Options &options)
{
  options.add_options()
      ("host-stats", "Enable displaying host stats", cxxopts::value<bool>()->default_value("false"))("print-to-terminal", "When set will print to terminal not overlay video", cxxopts::value<bool>()->default_value("false"));
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
DataAggregator *setup_hailo_data_aggregator(GstElement *pipeline, GMainLoop *main_loop, cxxopts::ParseResult result)
{
  gboolean print_fps = result["show-fps"].as<bool>();
  gboolean print_hailo_stats = result["hailo-stats"].as<bool>();
  gboolean print_host_stats = result["host-stats"].as<bool>();
  gboolean print_to_terminal = result["print-to-terminal"].as<bool>();

  // get process id
  pid_t pid = getpid();
  std::cout << "Parent process id: " << pid << std::endl;
  
  // get new data aggregator
  DataAggregator *data_aggregator = new DataAggregator();

  // set agg_data
  AggData *agg_data = new AggData();
  agg_data->pid = pid;

  // Set the pipeline element
  agg_data->pipeline = pipeline;

  // Set the main loop element
  agg_data->main_loop = main_loop;

  // Set additional options
  agg_data->print_fps = print_fps;
  agg_data->print_hailo_stats = print_hailo_stats;
  agg_data->print_host_stats = print_host_stats;
  agg_data->print_to_terminal = print_to_terminal;
  agg_data->print_required = print_fps || print_hailo_stats || print_host_stats;

  data_aggregator->initilize_agg_data(agg_data);
  // set callbacks
  if (agg_data->print_fps)
  {
    try
    {
      // set fps-measurements signal callback to print the measurements
      std::cout << "Setting fps-measurements signal callback" << std::endl;
      GstElement *display_0 = gst_bin_get_by_name(GST_BIN(agg_data->pipeline), "hailo_display");
      g_signal_connect(display_0, "fps-measurements", G_CALLBACK(fps_measurements_callback), data_aggregator);
    }
    catch (const std::exception &e)
    {
      std::cout << "Could not set fps-measurements signal callback make sure your display element name is hailo_display" << std::endl;
    }
  }

  if (agg_data->print_required)
  {
    // set timer to print stats
    g_timeout_add_seconds(1, (GSourceFunc)display_stats_callback, data_aggregator);
  }

  if (agg_data->print_host_stats)
  {
    // set timer to update host stats
    g_timeout_add_seconds(1, (GSourceFunc)update_host_stats_callback, data_aggregator);
  }

  if (agg_data->print_hailo_stats)
  {
    // set timer to update hailo stats
    std::cout << "Setting hailo stats" << std::endl;
    try
    {
      GstElement *hailostats = gst_bin_get_by_name(GST_BIN(agg_data->pipeline), "hailo_stats");
      if (hailostats == nullptr)
      {
        throw std::runtime_error("Could not set hailo stats make sure your hailodevicestats element name is hailo_stats");
      }
    }
    catch (const std::exception &e)
    {
      std::cout << "Error: " << e.what() << std::endl;
      std::cout << "You should add this to your pipeline: hailodevicestats name=hailo_stats silent=false " << std::endl;
    }
  }
  // Set the text overlay element
  if (not agg_data->print_to_terminal)
  {
    try
    {
      GstElement *text_overlay = gst_bin_get_by_name(GST_BIN(agg_data->pipeline), "text_overlay");
      if (text_overlay == nullptr)
      {
        throw std::runtime_error("Could not get text_overlay element make sure you have text_overlay element with the name text_overlay. Otherwise you can use the --print-to-terminal option to print to the terminal.");
      }
      agg_data->text_overlay = text_overlay;
    }
    catch (const std::exception &e)
    {
      std::cout << "Error: " << e.what() << std::endl;
      std::cout << "You should add this to your pipeline: textoverlay name=text_overlay" << std::endl;
    }
  }
  return data_aggregator;
}
