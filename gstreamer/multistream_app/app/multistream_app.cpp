/**
* Copyright 2023 Hailo Technologies Ltd.
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
* 
*     http://www.apache.org/licenses/LICENSE-2.0
* 
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
**/

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

// Open source includes
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

// Hailo app cpp utils include
#include "hailo_app_data_aggregator.hpp" // if used should be included before hailo_app_cpp_utils.hpp
#include "hailo_app_cpp_utils.hpp"
#include "hailo_app_useful_funcs.hpp"


#include "SrcBin.hpp"

GST_DEBUG_CATEGORY (app_debug);
#define GST_CAT_DEFAULT app_debug

// use --gst-debug=*debug:5 to enable debug messages
// use --fps-probe=true to enable fps probe and debug level INFO to get timestamps
struct AppData {
    GstElement* pipeline;
    GMainLoop* main_loop;
    // a vector of src bins
    std::vector<SrcBin*> src_bins;
};

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
const std::string STREAM_ID_SO = TAPPAS_WORKSPACE + "/apps/h8/gstreamer/libs/post_processes/libstream_id_tool.so";
const std::string HEF_PATH = TAPPAS_WORKSPACE + "/apps/h8/gstreamer/resources/hef/yolov5m_wo_spp_60p.hef";
const std::string QUEUE = "queue leaky=no max-size-bytes=0 max-size-time=0 ";


// sudo ifconfig enx08920489ee65 192.168.0.100 netmask 255.255.255.0
// const std::string RTSP_SRC_0 = "rtsp://192.168.0.101/axis-media/media.amp/?h264x=4 user-id=root user-pw=hailo";
// const std::string RTSP_SRC_1 = "rtsp://192.168.0.102/axis-media/media.amp/?h264x=4 user-id=root user-pw=hailo";

const std::string RTSP_SRC_0 = "rtsp://192.168.0.101/axis-media/media.amp/?h264x=4";
const std::string RTSP_SRC_1 = "rtsp://192.168.0.102/axis-media/media.amp/?h264x=4";
const std::string RTSP_SRC_2 = "rtsp://192.168.0.103/axis-media/media.amp/?h264x=4";
const std::string RTSP_SRC_3 = "rtsp://192.168.0.104/axis-media/media.amp/?h264x=4";
const std::string RTSP_SRC_4 = "rtsp://192.168.0.105/axis-media/media.amp/?h264x=4";


const std::string URI_SRC_0 = "file:///local/workspace/tappas/apps/h8/gstreamer/resources/mp4/detection0.mp4";
const std::string URI_SRC_1 = "file:///local/workspace/tappas/apps/h8/gstreamer/resources/mp4/detection1.mp4";
const std::string URI_SRC_2 = "file:///local/workspace/tappas/apps/h8/gstreamer/resources/mp4/detection2.mp4";
const std::string URI_SRC_3 = "file:///local/workspace/tappas/apps/h8/gstreamer/resources/mp4/detection3.mp4";
const std::string URI_SRC_4 = "file:///local/workspace/tappas/apps/h8/gstreamer/resources/mp4/detection4.mp4";


static GstPadProbeReturn pad_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer data)
{
    GstEvent *event = GST_PAD_PROBE_INFO_EVENT(info);
    if ((info->type & GST_PAD_PROBE_TYPE_EVENT_BOTH))
    {
        // print event type
        // GST_INFO("Event type: %s\n", GST_EVENT_TYPE_NAME(event));
        
        // if this is an EOS event
        if (GST_EVENT_TYPE(event) == GST_EVENT_EOS)
        {
            GST_INFO("EOS event received on source\n");
        }
        // if this is a segment event
        if (GST_EVENT_TYPE(event) == GST_EVENT_SEGMENT)
        {
            // get the segment event
            GstSegment *segment;
            gst_event_parse_segment(event, (const GstSegment **)&segment);
            GST_INFO("Segment event received on source \n");
            GST_INFO("Segment base: %" GST_TIME_FORMAT "\n", GST_TIME_ARGS(segment->base));
            GST_INFO("Segment start: %" GST_TIME_FORMAT "\n", GST_TIME_ARGS(segment->start));
            GST_INFO("Segment stop: %" GST_TIME_FORMAT "\n", GST_TIME_ARGS(segment->stop));
            GST_INFO("Segment time: %" GST_TIME_FORMAT "\n", GST_TIME_ARGS(segment->time));
            GST_INFO("Segment position: %" GST_TIME_FORMAT "\n", GST_TIME_ARGS(segment->position));
            GST_INFO("Segment duration: %" GST_TIME_FORMAT "\n", GST_TIME_ARGS(segment->duration));
            GST_INFO("Segment format: %s\n", gst_format_get_name(segment->format));
            GST_INFO("Segment flags: %d\n", segment->flags);
            GST_INFO("Segment rate: %f\n", segment->rate);
            GST_INFO("Segment applied_rate: %f\n", segment->applied_rate);
        }
        
    }
    return GST_PAD_PROBE_OK;
}


std::string create_sources(int num_of_src, std::string src_names[], bool use_rtsp=true, AppData* app_data=nullptr) {
    std::string result = "";
    for (int n = 0; n < num_of_src; n++) {
        // create the src bins
        if (!use_rtsp) {
            // input source is a video file
            // Create the SrcBin and add it to the vector
            app_data->src_bins.push_back(new SrcBin(SrcBin::SrcType::URI, src_names[n]));
        } else {
            // input source is a rtsp stream
            // Create the SrcBin and add it to the vector
            app_data->src_bins.push_back(new SrcBin(SrcBin::SrcType::RTSP, src_names[n]));
        }
        // create additional pipeline elements
        result += QUEUE + "name=src_bin_out_q_" + std::to_string(n) + " max-size-buffers=3 ! ";
        // result += "videorate name=videorate_" + std::to_string(n) + " ! video/x-raw,framerate=25/1 ! ";
        result += "queue leaky=downstream max-size-bytes=0 max-size-time=0 name=videorate_q_" + std::to_string(n) + " max-size-buffers=3 ! ";
        result += "hailofilter name=set_id_" + std::to_string(n) + " so-path=" + STREAM_ID_SO + " config-path=SRC_" + std::to_string(n) + " qos=false ! ";
        // result += "identity name=fps_probe_src_" + std::to_string(n) + " ! ";
        result += "videoscale name=videoscale_src_" + std::to_string(n) + " ! ";
        // ADD QUEUE
        result += QUEUE + "name=scale_q_" + std::to_string(n) + " max-size-buffers=3 ! ";
        // Create the roundroubin sink element
        result += "roundroubin.sink_" + std::to_string(n) + " ";
    }
    return result;
}

std::string create_sid_comp_pipelines(int num_of_src, std::string sync_pipeline="false")
{
    std::string  result = "hailostreamrouter name=sid ";
    for (int n = 0; n < num_of_src; n++) {
        result += " src_" + std::to_string(n) + "::input-streams=\"<sink_" + std::to_string(n) + ">\" ";
    }
    for (int n = 0; n < num_of_src; n++) {
        result += "sid.src_" + std::to_string(n) + " ! ";
        result += QUEUE + "name=sid_q_" + std::to_string(n) + " max-size-buffers=3 ! ";
        // result += "identity name=fps_probe_disp_" + std::to_string(n) + " ! ";
        // //result += "fpsdisplaysink video-sink=xvimagesink name=hailo_display_" + std::to_string(n) + " sync=false text-overlay=false ";
        result += "fpsdisplaysink name=hailo_display_" + std::to_string(n) + " sync=" + sync_pipeline + " ";
    }
    return result;
}

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
std::string create_pipeline_string(cxxopts::ParseResult result, AppData* app_data=nullptr)
{
    std::string stats_pipeline = "";
    std::string pipeline_string = "";
    std::string video_sink_element = "xvimagesink";
    std::string sync_pipeline = "false";
    int num_of_src = result["num-of-src"].as<int>();
    bool use_rtsp = false;
    std::vector<std::string> src_names;
    // If required add hailodevicestats to pipeline
    if (result["hailo-stats"].as<bool>()) {
        stats_pipeline = "hailodevicestats name=hailo_stats silent=false ";
    }
    if (result["sync-pipeline"].as<bool>()) {
        sync_pipeline = "true";
    }
    if (result["rtsp-src"].as<bool>()) {
        use_rtsp = true;
        src_names = {RTSP_SRC_0, RTSP_SRC_1, RTSP_SRC_2, RTSP_SRC_3, RTSP_SRC_4};
    } else {
        use_rtsp = false;
        src_names = {URI_SRC_0, URI_SRC_1, URI_SRC_2, URI_SRC_3, URI_SRC_4};
    }
    // convert result["rr-mode"] from int to string
    std::string roundrobin_mode = std::to_string(result["rr-mode"].as<int>());

    // Create the pipeline string
    pipeline_string += create_sources(num_of_src, src_names.data(), use_rtsp, app_data);
    pipeline_string += "hailoroundrobin name=roundroubin mode=" + roundrobin_mode + " ! ";
    pipeline_string += QUEUE + " name=roundrobin_q max-size-buffers=3 ! ";
    pipeline_string += "identity sync=true ! ";
    pipeline_string += QUEUE + " name=buf_q max-size-buffers=30 ! ";
    pipeline_string += "videoconvert name=preproc_convert qos=false n-threads=3 ! ";
    pipeline_string += QUEUE + " name=convert_q max-size-buffers=3 ! ";
    pipeline_string += "identity name=fps_probe_inference sync=true ! ";
    pipeline_string += "videoscale ! ";
    pipeline_string += "video/x-raw,width=640,height=640,pixel-aspect-ratio=1/1,format=RGB ! ";
    pipeline_string += QUEUE + " name=preproc_q max-size-buffers=3 ! ";
    pipeline_string += "hailonet hef-path=" + HEF_PATH + " ! ";
    pipeline_string += QUEUE + " name=postroc_q max-size-buffers=3 ! ";
    pipeline_string += "hailofilter name=filter_post so-path=" + POSTPROCESS_SO + " config-path=" + JSON_CONFIG_PATH + " qos=false ! ";
    // pipeline_string += "identity name=data_probe ! ";
    pipeline_string += QUEUE + " name=overlay_q max-size-buffers=3 ! ";
    pipeline_string += "hailooverlay qos=false ! ";
    pipeline_string += QUEUE + " name=display_convert_q max-size-buffers=3 ! ";
    pipeline_string += "videoconvert name=display_convert n-threads=3 ! ";
    pipeline_string += "textoverlay name=text_overlay ! ";
    pipeline_string += create_sid_comp_pipelines(num_of_src, sync_pipeline);
    pipeline_string += stats_pipeline;
    //print pipeline string
    GST_ERROR("Pipeline string: %s\n", pipeline_string.c_str());
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
    gst_init(&argc, &argv); // Initialize Gstreamer
    
    // Initialize the app debug category
    GST_DEBUG_CATEGORY_INIT (app_debug, "app_debug", 1, "My app debug category");
    // build argument parser
    cxxopts::Options options = build_arg_parser();
    // add custom options
    options.add_options()
    ("fps-probe", "Enables fps probes", cxxopts::value<bool>()->default_value("false"))
    ("rtsp-src", "Use RTSP sources", cxxopts::value<bool>()->default_value("false"))
    ("n, num-of-src", "Number of sources", cxxopts::value<int>()->default_value("2"))
    ("rr-mode", "Hailoroundrobin mode", cxxopts::value<int>()->default_value("2"))
    ("dump-dot-files", "Enables dumping of dot files", cxxopts::value<bool>()->default_value("false"));
    
    //add aggregator options
    //add_aggregator_options(options);

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

    // create app data struct
    AppData app_data;
    // Create the pipeline
    std::string pipeline_string = create_pipeline_string(result, &app_data);
    
    // Parse the pipeline string and create the pipeline
    GError *err = nullptr;
    GstElement *pipeline = gst_parse_launch(pipeline_string.c_str(), &err);
    checkErr(err);

    //connect source bins to the pipeline
    int num_of_src = result["num-of-src"].as<int>();
    for (int n = 0; n < num_of_src; n++) {
        // get the source element
        GstElement *src_elem = app_data.src_bins[n]->get();
        // add the source element to the pipeline
        gst_bin_add(GST_BIN(pipeline), src_elem);
        // get the source bin out queue
        GstElement *src_bin_out_q = gst_bin_get_by_name(GST_BIN(pipeline), ("src_bin_out_q_" + std::to_string(n)).c_str());
        // link the source element to the source bin out queue
        gst_element_link(src_elem, src_bin_out_q);
    }

    // Get the bus
    bus = gst_element_get_bus(pipeline);

    // Run hailo utils setup for basic run
    UserData user_data; // user data to pass to callbacks
    setup_hailo_utils(pipeline, bus, main_loop, &user_data, result);
    
    // Setup aggregator
    //user_data.data_aggregator = setup_hailo_data_aggregator(pipeline, main_loop, result);
    
    // Add probe examples
    if (result["fps-probe"].as<bool>())
    {
        g_print("Enabling fps probe\n");
        // get fps_probe element by name get all elements which start with fps_probe
        std::string element_prefix = "fps_probe";
        // Iterate over all elements in the pipeline recursively
        GstIterator* it = gst_bin_iterate_recurse(GST_BIN(pipeline));
        GValue item = G_VALUE_INIT;
        while (gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
            GstElement* element = GST_ELEMENT(g_value_get_object(&item));
            // Check if the element name starts with the specified prefix
            std::string element_name = GST_ELEMENT_NAME(element);
            if (element_name.compare(0, element_prefix.length(), element_prefix) == 0) {
                // Element name starts with the prefix
                g_signal_connect(element, "handoff", G_CALLBACK(fps_probe_callback), &user_data);
                // g_signal_connect(element, "handoff", G_CALLBACK(timestamp_probe_callback), &user_data);
            }
            g_value_unset(&item);
        }
        gst_iterator_free(it);
    }

    if (result["dump-dot-files"].as<bool>()) {
        g_print("Dumping dot files\n");
        // Dump the DOT file after the pipeline has been created this is before caps negotiation
        gst_debug_bin_to_dot_file(GST_BIN(pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "pipeline_before_negotiation");
    }
    
    // Set the pipeline state to PLAYING
    g_print("Setting pipeline to PLAYING");
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    
    //wait for the pipeline to finish state change
    GstStateChangeReturn ret = gst_element_get_state(pipeline, NULL, NULL, GST_CLOCK_TIME_NONE);

    if (ret == GST_STATE_CHANGE_FAILURE) {
        GST_ERROR("Failed to start pipeline");
        exit(1);
    }

    // Set watchdog timers for each source (RTSP only) and start bus sync handler
    for (int n = 0; n < num_of_src; n++) {
        if (app_data.src_bins[n]->type == SrcBin::SrcType::RTSP) {
            app_data.src_bins[n]->start_watchdog_thread();
        }
        // app_data.src_bins[n]->start_bus_sync_handler();
    }
    
    // gst_bus_set_sync_handler(bus, (GstBusSyncHandler)SrcBin::bus_sync_handler, NULL, NULL);
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
