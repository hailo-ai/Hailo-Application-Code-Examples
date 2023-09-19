/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
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
#include <iostream>
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
#include "hailo_app_cpp_utils.hpp"
//******************************************************************
// Pipeline Macros
//******************************************************************
const std::string APP_RUNTIME_DIR = getexepath();

// Get TAPPAS_WORKSPACE from environment variable
const char* temp = std::getenv("TAPPAS_WORKSPACE");
const std::string TAPPAS_WORKSPACE = temp ? std::string(temp) : "";

const std::string RESOURCES_DIR = APP_RUNTIME_DIR + "/resources";
const std::string POSTPROCESS_DIR = TAPPAS_WORKSPACE + "/apps/h8/gstreamer/libs/post_processes/";
const std::string POSTPROCESS_SO = POSTPROCESS_DIR + "/libyolo_post.so";
const std::string HEF_PATH = RESOURCES_DIR + "/yolov5s_vehicles_nv12.hef";
const std::string CAR_DETECTION_JSON_CONFIG_PATH = RESOURCES_DIR + "/configs/yolov5_vehicle_detection.json";
const std::string CROPPING_ALGORITHMS_DIR = POSTPROCESS_DIR + "/cropping_algorithms";
const std::string WHOLE_BUFFER_CROP_SO = CROPPING_ALGORITHMS_DIR + "/libwhole_buffer.so";
const std::string DEFAULT_VDEVICE_KEY="1";
// License Plate Detection Macros
const std::string LICENSE_PLATE_DETECTION_HEF = RESOURCES_DIR + "/tiny_yolov4_license_plates_nv12.hef";
const std::string LICENSE_PLATE_DETECTION_POST_SO = POSTPROCESS_DIR +  "/libyolo_post.so";
const std::string LICENSE_PLATE_DETECTION_POST_FUNC = "tiny_yolov4_license_plates";
const std::string LICENSE_PLATE_DETECTION_JSON_CONFIG_PATH = RESOURCES_DIR + "/configs/yolov4_licence_plate.json";
// License Plate OCR Macros
const std::string LICENSE_PLATE_OCR_HEF = RESOURCES_DIR + "/lprnet_yuy2.hef";
const std::string LICENSE_PLATE_OCR_POST_SO = POSTPROCESS_DIR + "/libocr_post.so";
const std::string LPR_OCR_SINK = TAPPAS_WORKSPACE + "/apps/h8/gstreamer/libs/apps/license_plate_recognition/" + "liblpr_ocrsink.so";

// Cropping Algorithm Macros
const std::string LICENSE_PLATE_CROP_SO = TAPPAS_WORKSPACE + "/apps/h8/gstreamer/libs/post_processes/cropping_algorithms/" + "liblpr_croppers.so";
const std::string LICENSE_PLATE_DETECTION_CROP_FUNC = "vehicles_without_ocr";
const std::string LICENSE_PLATE_OCR_CROP_FUNC = "license_plate_quality_estimation";

// check_resources function that checks if all resources exist
void check_resources()
{
    std::pair<std::string, std::string> resources[] = {
        {"TAPPAS_WORKSPACE", TAPPAS_WORKSPACE},
        {"POSTPROCESS_SO", POSTPROCESS_SO},
        {"LICENSE_PLATE_DETECTION_HEF", LICENSE_PLATE_DETECTION_HEF},
        {"LICENSE_PLATE_DETECTION_POST_SO", LICENSE_PLATE_DETECTION_POST_SO},
        {"LICENSE_PLATE_OCR_HEF", LICENSE_PLATE_OCR_HEF},
        {"LICENSE_PLATE_OCR_POST_SO", LICENSE_PLATE_OCR_POST_SO},
        {"LICENSE_PLATE_CROP_SO", LICENSE_PLATE_CROP_SO}
    };

    for (auto& resource : resources)
    {
        std::ifstream f(resource.second.c_str());
        if (!f.good())
        {
            std::cout << "ERROR: " << resource.first << " not found" << std::endl;
            exit(0);
        }
    }
}
    
std::string DECODER_FORMAT="NV12";
const std::string QUEUE = "queue leaky=no max-size-bytes=0 max-size-time=0 ";

std::string create_compositor_locations(int num_of_src, int width, int height, int row_size)
{
    std::string compositor_locations = "";
    int xpos = 0;
    int ypos = 0;
    for (int i = 0; i < num_of_src; i++)
    {
        compositor_locations += "sink_" + std::to_string(i) + "::xpos=" + std::to_string(xpos) + " sink_" + std::to_string(i) + "::ypos=" + std::to_string(ypos) + " ";
        xpos += width;
        if (xpos >= width * row_size)
        {
            xpos = 0;
            ypos += height;
        }
    }
    return compositor_locations;
}


//******************************************************************
// PIPELINE CREATION
//******************************************************************
std::string create_pipeline_string(cxxopts::ParseResult result)
{
    int num_of_src = result["num-of-inputs"].as<int>();
    bool debug_mode = result["debug"].as<bool>();
    bool enable_display = result["enable-display"].as<bool>();
    bool lpd_display = result["lpd-display"].as<bool>();
    int num_of_buffers = result["num-of-buffers"].as<int>();
    int width, height, streams_in_line;

    std::string pipeline_string = "";
    std::string stats_pipeline = ""; //hailodevicestats name=hailo_stats silent=false ";
    std::string debug_tracker = "";
    std::string video_sink_element = "fakesink";
    std::string internal_offset = "true"; // should be false when using streaming source TBD
    std::string sync_pipeline = "false";
    if (enable_display && (num_of_src == 1))
        sync_pipeline = "true";
    // If required add hailodevicestats to pipeline
    if (result["hailo-stats"].as<bool>()) {
        stats_pipeline = "hailodevicestats name=hailo_stats silent=false ";
    }
    
    // source pipeline
    std::string sources;
    std::string streamrouter_input_streams;
    int start_index = 0;
    std::string vaapi_res = "";
    if(num_of_src > 11){
        width = 640;
        height = 360;
        streams_in_line = 4;
    } else if(num_of_src > 4){
        width = 640;
        height = 360;
        streams_in_line = 3;
    }
    else{
        width = 960;
        height = 540;
        streams_in_line = 2;
    }
    std::string compositor_locations = create_compositor_locations(num_of_src,width, height, streams_in_line);

    std::string car_det_net_params = " batch-size=8 vdevice-key=1 scheduling-algorithm=1 scheduler-threshold=1 scheduler-timeout-ms=100 ";
    std::string lp_det_net_params = " batch-size=8 vdevice-key=1 scheduling-algorithm=1 scheduler-threshold=8 scheduler-timeout-ms=100 ";
    std::string ocr_net_params = " batch-size=16 vdevice-key=1 scheduling-algorithm=1 scheduler-threshold=8 scheduler-timeout-ms=200 ";
    
    if (debug_mode){
        vaapi_res = ",width=1920,height=1080 ! ";
        debug_tracker = " debug=true ";
    }
    else{
        vaapi_res = ",width=1920,height=1080 ! ";
    }
    if (enable_display){
        video_sink_element = "xvimagesink";
    }
    else{
        video_sink_element = "fakesink";
        // video_sink_element = "xvimagesink";
    }
    for (int n = start_index; n < num_of_src; ++n)
    {
        sources += "filesrc num-buffers=" + std::to_string(num_of_buffers) + " location=" + RESOURCES_DIR + "/videos/lpr_video" + std::to_string(n) + ".mp4 name=source_" + std::to_string(n) + " ! ";
        sources += "queue name=pre_decode_q_" + std::to_string(n) + " leaky=no max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! ";
        sources += "qtdemux ! ";
        sources += QUEUE + " name=pre_decode_qtdemux_q_" + std::to_string(n) + " max-size-buffers=5 ! ";
        sources += "decodebin ! ";
        sources += "queue name=pre_postproc_q_" + std::to_string(n) + " leaky=no max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! ";
        sources += "videoconvert ! ";
        sources += "video/x-raw,format=" + DECODER_FORMAT + vaapi_res;
        sources += "queue name=hailo_prestream_mux_q_" + std::to_string(n) + " leaky=no max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! ";
        
        sources += "fun.sink_" + std::to_string(n) + " ";
        if (enable_display)
        {
            sources += "sid.src_" + std::to_string(n) + " ! ";
            sources += "queue name=pre_comp_videoconvert_q_" + std::to_string(n) + " leaky=no max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! ";
            sources += "videoconvert name=pre_comp_videoconvert_" + std::to_string(n) + " qos=false ! ";
            sources += "queue name=comp_q_" + std::to_string(n) + " leaky=no max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! ";
            sources += "comp.sink_" + std::to_string(n) + " ";
        }
        streamrouter_input_streams += " src_" + std::to_string(n) + "::input-streams=\"<sink_" + std::to_string(n) + ">\" ";
    }
    
    pipeline_string = sources + " hailoroundrobin name=fun ! ";
    pipeline_string += QUEUE + " name=hailo_pre_infer_q_0 max-size-buffers=30 ! ";
    pipeline_string += QUEUE + " name=car_det_pre_crop max-size-buffers=30 ! ";
    pipeline_string += "hailocropper so-path=" + WHOLE_BUFFER_CROP_SO + " use-letterbox=false function-name=create_crops internal-offset=" + internal_offset + " name=car_detect_cropper ";
    
    pipeline_string += "hailoaggregator name=car_detect_agg ";
    pipeline_string += "car_detect_cropper.src_0 ! queue name=car_detect_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! ";
    

    pipeline_string += "car_detect_agg.sink_0 "; 
    pipeline_string += "car_detect_cropper.src_1 ! ";
    // car detect pipeline
    pipeline_string += "queue name=hailo_car_detect_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! ";
    pipeline_string += "video/x-raw,width=1920,height=1080,pixel-aspect-ratio=1/1 ! ";
    pipeline_string += "videoscale n-threads=4 method=0 add-borders=false qos=false ! ";
    pipeline_string += "video/x-raw,width=640,height=640,pixel-aspect-ratio=1/1 ! ";
    pipeline_string += QUEUE + " name=car_det_format_q3 max-size-buffers=3 ! ";
    if (false){
        pipeline_string += "tee name=det_tee ! ";
        pipeline_string += "queue name=det_q_overlay leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! ";
        pipeline_string += "videoconvert ! xvimagesink name=det_xvimagesink sync=false async=true qos=false ";
        pipeline_string += "det_tee. ! ";
    }   
    pipeline_string += "hailonet hef-path=" + HEF_PATH + car_det_net_params + " ! ";
    pipeline_string += "queue name=hailo_post_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! ";
    pipeline_string += "hailofilter name=filter_post so-path=" + POSTPROCESS_SO + " config-path=" + CAR_DETECTION_JSON_CONFIG_PATH + " qos=false ! ";
    pipeline_string += "queue name=hailo_remap_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! ";
    pipeline_string += "hailotracker name=hailo_tracker keep-past-metadata=true kalman-dist-thr=.7 iou-thr=.9 keep-tracked-frames=2 keep-lost-frames=2 class-id=1 qos=false " + debug_tracker + " ! ";
    pipeline_string += "queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! ";
    
    
    pipeline_string += "car_detect_agg.sink_1 ";
    pipeline_string += "car_detect_agg. ! "; 
    pipeline_string += "identity name=fps_probe ! ";
    
    if (enable_display) {
        pipeline_string += "tee name=display_tee ! ";
        pipeline_string += "queue name=display_q_1 leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! ";
        if (num_of_src > 1) {
            pipeline_string += "videoscale ! video/x-raw,width=" + std::to_string(width) + ",height=" + std::to_string(height) + " ! ";
        }
        pipeline_string += "queue name=display_q_2 leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! ";
        pipeline_string += "hailooverlay qos=false ! ";
        pipeline_string += "queue name=display_q_3 leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! ";
        pipeline_string += "hailostreamrouter name=sid " + streamrouter_input_streams + " ";
        pipeline_string += "compositor name=comp start-time-selection=0 " + compositor_locations + " ! ";
        pipeline_string += "textoverlay name=text_overlay ! ";
        pipeline_string += "queue name=display_q_4 leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! ";
        pipeline_string += "fpsdisplaysink video-sink=" + video_sink_element + " name=hailo_display sync=" + sync_pipeline + " text-overlay=false signal-fps-measurements=true ";
        pipeline_string += "display_tee. ! ";
        pipeline_string += "queue name=pre_lpd_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! ";
    }

    // License plate pipeline
    std::string license_plate_detection_pipeline = "";
    license_plate_detection_pipeline += "hailocropper so-path=" + LICENSE_PLATE_CROP_SO + " function-name=" + LICENSE_PLATE_DETECTION_CROP_FUNC \
    + " internal-offset=" + internal_offset + " drop-uncropped-buffers=true use-letterbox=false name=license_plate_detect_cropper ";
    license_plate_detection_pipeline += "hailoaggregator name=lpd_agg ";
    license_plate_detection_pipeline += "license_plate_detect_cropper.src_0 ! ";
    license_plate_detection_pipeline += "queue name=lpd_q_bypass leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! ";
    license_plate_detection_pipeline += "video/x-raw,format=NV12,width=1920,height=1080 ! ";
    license_plate_detection_pipeline += "lpd_agg.sink_0 ";
    license_plate_detection_pipeline += "license_plate_detect_cropper.src_1 ! ";
    license_plate_detection_pipeline += "video/x-raw,width=416,height=416,pixel-aspect-ratio=1/1,format=NV12 ! ";
    license_plate_detection_pipeline += "queue name=lpd_q_1 leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! ";
    if (false){
        license_plate_detection_pipeline += "tee name=car_tee ! ";
        license_plate_detection_pipeline += "queue name=car_q_overlay leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! ";
        license_plate_detection_pipeline += "videoconvert ! xvimagesink name=car_xvimagesink sync=false async=true qos=false ";
        license_plate_detection_pipeline += "car_tee. ! ";
    }
    license_plate_detection_pipeline += "hailonet hef-path=" + LICENSE_PLATE_DETECTION_HEF + lp_det_net_params + " ! ";
    license_plate_detection_pipeline += "queue name=lpd_q_2 leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! ";
    license_plate_detection_pipeline += "hailofilter so-path=" + LICENSE_PLATE_DETECTION_POST_SO + " config-path=" + LICENSE_PLATE_DETECTION_JSON_CONFIG_PATH + " function-name=" + LICENSE_PLATE_DETECTION_POST_FUNC + " qos=false ! ";
    license_plate_detection_pipeline += "queue name=lpd_q_3 leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! ";
    license_plate_detection_pipeline += "lpd_agg.sink_1 ";
    license_plate_detection_pipeline += "lpd_agg. ! ";
    pipeline_string += license_plate_detection_pipeline;
    
    // License plate OCR pipeline
    std::string license_plate_ocr_pipeline = "";
    
    license_plate_ocr_pipeline += "hailocropper so-path=" + LICENSE_PLATE_CROP_SO + " function-name=" + LICENSE_PLATE_OCR_CROP_FUNC \
    + " internal-offset=" + internal_offset + " drop-uncropped-buffers=true use-letterbox=false name=license_plate_ocr_cropper ";
    license_plate_ocr_pipeline += "hailoaggregator name=lpd_ocr_agg ";
    license_plate_ocr_pipeline += "license_plate_ocr_cropper.src_1 ! ";
    license_plate_ocr_pipeline += "video/x-raw,width=304,height=75,pixel-aspect-ratio=1/1,format=NV12 ! ";
    license_plate_ocr_pipeline += "queue name=ocr_q_1 leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! ";
    license_plate_ocr_pipeline += "videoconvert name=videoconvert_ocr ! ";
    license_plate_ocr_pipeline += "video/x-raw,width=304,height=75,pixel-aspect-ratio=1/1,format=YUY2 ! ";
    if (lpd_display){
        license_plate_ocr_pipeline += "tee name=lpd_tee ! ";
        license_plate_ocr_pipeline += "queue name=lpd_q_overlay leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! ";
        license_plate_ocr_pipeline += "videoconvert ! xvimagesink name=lpd_xvimagesink sync=false async=true qos=false ";
        license_plate_ocr_pipeline += "lpd_tee. ! ";
    }
    license_plate_ocr_pipeline += "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! ";
    license_plate_ocr_pipeline += "hailonet hef-path=" + LICENSE_PLATE_OCR_HEF + ocr_net_params + " ! ";
    license_plate_ocr_pipeline += "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! ";
    license_plate_ocr_pipeline += "hailofilter so-path=" + LICENSE_PLATE_OCR_POST_SO + " qos=false ! ";
    
    license_plate_ocr_pipeline += "lpd_ocr_agg.sink_1 ";
    license_plate_ocr_pipeline += "license_plate_ocr_cropper.src_0 ! ";
    license_plate_ocr_pipeline += "queue leaky=no max-size-buffers=50 max-size-bytes=0 max-size-time=0 ! ";
    license_plate_ocr_pipeline += "lpd_ocr_agg.sink_0 ";
    license_plate_ocr_pipeline += "lpd_ocr_agg. ! ";
    license_plate_ocr_pipeline += "queue leaky=no max-size-bytes=0 max-size-time=0 max-size-buffers=3 name=post_ocr_q ! ";
    license_plate_ocr_pipeline += "hailofilter use-gst-buffer=true so-path=" + LPR_OCR_SINK + " qos=false ! ";
    
    pipeline_string += license_plate_ocr_pipeline;
    pipeline_string += "fakesink name=license_plate_ocr_sink sync=false async=false qos=false ";

    pipeline_string += stats_pipeline;
    std::cout << "Pipeline:" << std::endl;
    std::cout << "gst-launch-1.0 " << pipeline_string << std::endl;
    // Combine and return the pipeline:
    return (pipeline_string);
}
//******************************************************************
// PROBE CALLBACK
//******************************************************************

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
    UserData* data = static_cast<UserData*>(user_data);
    HailoROIPtr roi;
    std::vector<HailoDetectionPtr> detections;
    std::vector<HailoDetectionPtr> sub_detections;
    roi = get_hailo_main_roi(buffer, true);
    detections = hailo_common::get_hailo_detections(roi);
    
    //print detections
    for (auto detection : detections)
    {
        auto bbox = detection->get_bbox();
        std::cout << "Detection: " << detection->get_label() << " " << std::to_string(bbox.width()) << " " << std::to_string(bbox.xmin()) << std::endl;
        sub_detections = hailo_common::get_hailo_detections(detection);
        for (auto sub_detection : sub_detections)
        {
            auto sub_bbox = sub_detection->get_bbox();
            std::cout << "Sub Detection: " << sub_detection->get_label() << " " << std::to_string(sub_bbox.width()) << " " << std::to_string(sub_bbox.xmin()) << std::endl;
        }
    }
}

//******************************************************************
// MAIN
//******************************************************************

int main(int argc, char *argv[])
{
    check_resources();
    // Set the GST_DEBUG_DUMP_DOT_DIR environment variable to dump a DOT file
    setenv("GST_DEBUG_DUMP_DOT_DIR", APP_RUNTIME_DIR.c_str(), 1);
     
    // build argument parser
    cxxopts::Options options = build_arg_parser();
    // add custom options
    options.add_options()
    ("n, num-of-inputs", "Number of inputs", cxxopts::value<int>()->default_value("4"))
    ("d, debug", "Enable debug", cxxopts::value<bool>()->default_value("false"))
    ("p, print-pipeline", "Only prints pipeline and exits", cxxopts::value<bool>()->default_value("false"))
    ("enable-display", "Enables display", cxxopts::value<bool>()->default_value("false"))
    ("lpd-display", "Enables license plate detection display", cxxopts::value<bool>()->default_value("false"))
    ("num-of-buffers","Number of buffers before EOS", cxxopts::value<int>()->default_value("-1"))
    ("dump-dot-files", "Enables dumping of dot files", cxxopts::value<bool>()->default_value("false"));
    ;

    // parse arguments
    auto result = options.parse(argc, argv);
    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    
    // Prepare pipeline components
    GstBus *bus;
    GMainLoop *main_loop;
    std::string src_pipeline_string;
    gst_init(&argc, &argv); // Initialize Gstreamer
    // Create the main loop
    main_loop = g_main_loop_new(NULL, FALSE);

    // Create the pipeline
    std::string pipeline_string = create_pipeline_string(result);

    // Parse the pipeline string and create the pipeline
    GError *err = nullptr;
    GstElement *pipeline = gst_parse_launch(pipeline_string.c_str(), &err);
    checkErr(err);
    
    // Get the bus
    bus = gst_element_get_bus(pipeline);

    // Run hailo utils setup for basic run
    UserData user_data;
    setup_hailo_utils(pipeline, bus, main_loop, &user_data, result);
    
    // get fps_probe element
    GstElement *fps_probe = gst_bin_get_by_name(GST_BIN(pipeline), "fps_probe");
    // set probe callback
    g_signal_connect(fps_probe, "handoff", G_CALLBACK(fps_probe_callback), &user_data);

    // Run the pipeline
    if (not result["print-pipeline"].as<bool>())
    {
        // Set the pipeline state to PLAYING
        std::cout << "Setting pipeline to PLAYING" << std::endl;
        gst_element_set_state(pipeline, GST_STATE_PLAYING);

        if (result["dump-dot-files"].as<bool>()) {
        g_print("Dumping dot files, adding delay to make sure state transition is done....\n");
        sleep(5);
        // Dump the DOT file after the pipeline has been set to PAUSED
        gst_debug_bin_to_dot_file(GST_BIN(pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "pipeline_paused");
        }
        // Run the main loop this is blocking will run until the main loop is stopped
        g_main_loop_run(main_loop);
    }
    // Free resources
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_deinit();
    gst_object_unref(pipeline);
    gst_object_unref(bus);
    g_main_loop_unref(main_loop);

    return 0;
}
