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

// Tappas includes
#include "hailo_objects.hpp"
#include "hailo_common.hpp"
#include "gst_hailo_meta.hpp"

// Hailo app cpp utils include
#include "hailo_app_data_aggregator.hpp"
#include "hailo_app_useful_funcs.hpp"

//******************************************************************
// Pipeline Macros
//******************************************************************
// get TAPPAS_WORKSPACE environment variable
const std::string TAPPAS_WORKSPACE = getenv("TAPPAS_WORKSPACE");
// get the app runtime directory
const std::string APP_RUNTIME_DIR = getexepath();
// Application specific macros
const std::string RESOURCES_DIR = APP_RUNTIME_DIR + "/resources";
const std::string JSON_CONFIG_PATH = RESOURCES_DIR + "/yolov5.json";
const std::string POSTPROCESS_SO = TAPPAS_WORKSPACE + "/apps/h8/gstreamer/libs/post_processes/libyolo_post.so";
const std::string HEF_PATH = TAPPAS_WORKSPACE + "/apps/h8/gstreamer/resources/hef/yolov5m_wo_spp_60p.hef";
const std::string QUEUE = "queue leaky=no max-size-bytes=0 max-size-time=0 ";

//******************************************************************
// PIPELINE CREATION
//******************************************************************
/**
 * @brief Create the pipeline string object
 *
 * @param cxxopts result
 *  input_src - std::string
 *      A video file path or usb camera name (/dev/video*)
 *  hailo-stats - bool
 *     If true, add hailodevicestats to pipeline
 * @return std::string
 *         The full pipeline string.
 */
std::string create_pipeline_string(cxxopts::ParseResult result)
{
    std::string stats_pipeline = "";
    std::string pipeline_string = "";
    std::string video_sink_element = "xvimagesink";
    std::string sync_pipeline = "false";
    
    // If required add hailodevicestats to pipeline
    if (result["hailo-stats"].as<bool>()) {
        stats_pipeline = "hailodevicestats name=hailo_stats silent=false ";
    }
    if (result["sync-pipeline"].as<bool>()) {
        sync_pipeline = "true";
    }
    // Check if input source is a video file or usb camera
    std::string input_src = result["input"].as<std::string>();
    // check regex for /dev/video*
    std::regex re("/dev/video[0-9]+");
    if (std::regex_match(input_src, re)) {
        // input source is a usb camera
        pipeline_string = "v4l2src device=" + input_src + " ! image/jpeg ! decodebin ! ";
        pipeline_string += QUEUE + " name=decode_q max-size-buffers=3 ! ";
        pipeline_string += "videoflip video-direction=horiz ! ";
    } else {
        // input source is a video file
        pipeline_string = "uridecodebin uri=" + input_src + " ! ";
        pipeline_string += QUEUE + " name=decode_q max-size-buffers=3 ! ";
    }
    pipeline_string += "videoconvert name=preproc_convert qos=false ! ";
    pipeline_string += QUEUE + " name=convert_q max-size-buffers=3 ! ";
    pipeline_string += "identity name=fps_probe ! ";
    pipeline_string += "videoscale ! ";
    pipeline_string += "video/x-raw,width=640,height=640,pixel-aspect-ratio=1/1,format=RGB ! ";
    pipeline_string += QUEUE + " name=preproc_q max-size-buffers=3 ! ";
    pipeline_string += "hailonet hef-path=" + HEF_PATH + " ! ";
    pipeline_string += QUEUE + " name=postroc_q max-size-buffers=3 ! ";
    pipeline_string += "hailofilter name=filter_post so-path=" + POSTPROCESS_SO + " config-path=" + JSON_CONFIG_PATH + " qos=false ! ";
    pipeline_string += "identity name=data_probe ! ";
    pipeline_string += QUEUE + " name=overlay_q max-size-buffers=3 ! ";
    pipeline_string += "hailooverlay qos=false ! ";
    pipeline_string += QUEUE + " name=display_convert_q max-size-buffers=3 ! ";
    pipeline_string += "videoconvert ! ";
    pipeline_string += "textoverlay name=text_overlay ! ";
    pipeline_string += "fpsdisplaysink video-sink=" + video_sink_element + " name=hailo_display sync=" + sync_pipeline + " text-overlay=false signal-fps-measurements=true ";
    pipeline_string += stats_pipeline;
    std::cout << "Pipeline:" << std::endl;
    std::cout << "gst-launch-1.0 " << pipeline_string << std::endl;
    // Combine and return the pipeline:
    return (pipeline_string);
}


//******************************************************************
// MAIN
//******************************************************************

int main(int argc, char *argv[])
{
    // Set the GST_DEBUG_DUMP_DOT_DIR environment variable to dump a DOT file
    setenv("GST_DEBUG_DUMP_DOT_DIR", APP_RUNTIME_DIR.c_str(), 1);
        
    // Prepare pipeline components
    GstBus *bus;
    GMainLoop *main_loop;
    GstStateChangeReturn ret;

    gst_init(&argc, &argv); // Initialize Gstreamer
   
    // build argument parser
    cxxopts::Options options = build_arg_parser();
    // add custom options
    options.add_options()
    ("p, print-pipeline", "Only prints pipeline and exits", cxxopts::value<bool>()->default_value("false"))
    ("probe-example", "Enables probe example", cxxopts::value<bool>()->default_value("false"))
    ("dump-dot-files", "Enables dumping of dot files", cxxopts::value<bool>()->default_value("false"));
    
    //add aggregator options
    add_aggregator_options(options);

    // parse arguments
    auto result = options.parse(argc, argv);
    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    // print the app runtime directory
    std::cout << "APP_RUNTIME_DIR: " << APP_RUNTIME_DIR << std::endl;
     
    // Create the main loop
    main_loop = g_main_loop_new(NULL, FALSE);

    // Create the pipeline
    std::string pipeline_string = create_pipeline_string(result);
    
    if (result["print-pipeline"].as<bool>())
    {
        std::cout << "Pipeline:" << std::endl;
        std::cout << "gst-launch-1.0 " << pipeline_string << std::endl;
        exit(0);
    }
    // Parse the pipeline string and create the pipeline
    GError *err = nullptr;
    GstElement *pipeline = gst_parse_launch(pipeline_string.c_str(), &err);
    if (err){
        GST_ERROR("Failed to create pipeline: %s", err->message);
        exit(1);
    }

    // Get the bus
    bus = gst_element_get_bus(pipeline);

    // Run hailo utils setup for basic run
    UserData user_data; // user data to pass to callbacks
    setup_hailo_utils(pipeline, bus, main_loop, &user_data, result);
    
    // Setup aggregator
    user_data.data_aggregator = setup_hailo_data_aggregator(pipeline, main_loop, result);
    
    // Add probe examples
    if (result["probe-example"].as<bool>())
    {
        // get fps_probe element by name based on identity handoff signal
        GstElement *fps_probe = gst_bin_get_by_name(GST_BIN(pipeline), "fps_probe");
        // set probe callback
        g_signal_connect(fps_probe, "handoff", G_CALLBACK(fps_probe_callback), &user_data);


        // get data_probe element by name (should probe after postprocessing is done)
        GstElement * data_probe = gst_bin_get_by_name(GST_BIN(pipeline), "data_probe");
        // set probe callback
        g_signal_connect(data_probe, "handoff", G_CALLBACK(detections_probe_callback), &user_data);
        
    }
    // Set the pipeline state to PAUSED
    std::cout << "Setting pipeline to PAUSED" << std::endl;
    gst_element_set_state(pipeline, GST_STATE_PAUSED);
    ret = gst_element_get_state(pipeline, NULL, NULL, 10 * GST_SECOND);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        g_error("Failed to move pipeline to PAUSED. Try running with --gst-debug=3\n");
        exit(1);
    } else if (ret == GST_STATE_CHANGE_NO_PREROLL) {
        g_info("Moved to PAUSED, no preroll. Expected in live sources\n");
    } else if (ret == GST_STATE_CHANGE_ASYNC) {
        g_warning("Pipeline reached timeout switching to paused\n");
    }

    if (result["dump-dot-files"].as<bool>()) {
        g_print("Dumping dot files, adding delay to make sure state transition is done....\n");
        sleep(10);
        // Dump the DOT file after the pipeline has been set to PAUSED
        gst_debug_bin_to_dot_file(GST_BIN(pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "pipeline_paused");
    }
    
    // Set the pipeline state to PLAYING
    std::cout << "Setting pipeline to PLAYING" << std::endl;
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    
    //wait for the pipeline to finish state change
    ret = gst_element_get_state(pipeline, NULL, NULL, 10 * GST_SECOND);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        g_error("Failed to move pipeline to PLAYING. Try running with --gst-debug=3\n");
        exit(1);
    } else if (ret == GST_STATE_CHANGE_ASYNC) {
        g_warning("Pipeline reached timoute switching to playing\n");
    }

    if (result["dump-dot-files"].as<bool>()) {
        // Dump the DOT file after the pipeline has been set to PLAYING
        gst_debug_bin_to_dot_file(GST_BIN(pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "pipeline_playing");
    }
    // Run the main loop this is blocking will run until the main loop is stopped
    g_main_loop_run(main_loop);
    
    // Free resources
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_deinit();
    gst_object_unref(pipeline);
    gst_object_unref(bus);
    g_main_loop_unref(main_loop);

    return 0;
}
