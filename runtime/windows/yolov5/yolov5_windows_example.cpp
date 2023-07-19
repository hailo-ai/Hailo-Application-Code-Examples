/**
 * Copyright 2020 (C) Hailo Technologies Ltd.
 * All rights reserved.
 *
 * Hailo Technologies Ltd. ("Hailo") disclaims any warranties, including, but not limited to,
 * the implied warranties of merchantability and fitness for a particular purpose.
 * This software is provided on an "AS IS" basis, and Hailo has no obligation to provide maintenance,
 * support, updates, enhancements, or modifications.
 *
 * You may use this software in the development of any project.
 * You shall not reproduce, modify or distribute this software without prior written permission.
 **/
/**
 * @file yolov5_windows_example.cpp
 * @brief This example demonstrates running inference with virtual streams on yolov5m
 **/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <future>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>


#include "yolov5_post_processing.hpp"
#include "double_buffer.hpp"

#include "common.h"
#include "hailo/hailort.h"

#define MAX_EDGE_LAYERS (16)
#define MAX_BATCH (64)
#define FRAME_WIDTH (640)
#define FRAME_HEIGHT (640)
#define PROCESSED_VID_FILE ("./processed_video.mp4")

cv::Mat* g_frame;
uint8_t *dst_data[MAX_EDGE_LAYERS][MAX_BATCH] = {NULL};
uint32_t g_TotalFrame = 0xFFFFFFFF;
extern float bbox_array[][6];
extern unsigned int box_index;

hailo_status build_streams(hailo_configured_network_group network_group,
    hailo_input_vstream *input_vstreams, size_t *input_frame_sizes,
    hailo_output_vstream *output_vstreams, size_t *output_frame_sizes, uint8_t *(*m_dst_data)[MAX_BATCH],
    size_t *num_output_streams)
{
    hailo_status status = HAILO_UNINITIALIZED;
    hailo_input_vstream_params_by_name_t input_vstream_params[MAX_EDGE_LAYERS];
    hailo_output_vstream_params_by_name_t output_vstream_params[MAX_EDGE_LAYERS];

    size_t input_vstream_size = 1;
    // Make sure it can hold amount of vstreams for hailo_make_input/output_vstream_params
    size_t output_vstream_size = MAX_EDGE_LAYERS;

    // prepare all input vstreams param data in advance
    status = hailo_make_input_vstream_params(network_group, true, HAILO_FORMAT_TYPE_AUTO,
        input_vstream_params, &input_vstream_size);
    REQUIRE_SUCCESS(status, l_exit, "Failed making input virtual stream params");

    // prepare all output vstreams param data in advance
    status = hailo_make_output_vstream_params(network_group, true, HAILO_FORMAT_TYPE_AUTO,
        output_vstream_params, &output_vstream_size);
    REQUIRE_SUCCESS(status, l_exit, "Failed making output virtual stream params");
    *num_output_streams = output_vstream_size;

    // create all input vstreams data in advance
    status = hailo_create_input_vstreams(network_group, input_vstream_params, input_vstream_size, input_vstreams);
    REQUIRE_SUCCESS(status, l_exit, "Failed creating virtual stream");

    // create all output vstreams data in advance
    status = hailo_create_output_vstreams(network_group, output_vstream_params, output_vstream_size, output_vstreams);
    REQUIRE_SUCCESS(status, l_release_input_vstream, "Failed creating virtual stream");

    for (size_t i = 0; i < input_vstream_size; i++)
    {
        status = hailo_get_input_vstream_frame_size(input_vstreams[i], &input_frame_sizes[i]);
        REQUIRE_SUCCESS(status, l_clear_buffers, "Failed getting input virtual stream frame size");
    }  

    for (size_t i = 0; i < output_vstream_size; i++)
    {
        status = hailo_get_output_vstream_frame_size(output_vstreams[i], &output_frame_sizes[i]);
        REQUIRE_SUCCESS(status, l_clear_buffers, "Failed getting input virtual stream frame size");

        for (uint8_t j = 0; j < MAX_BATCH; j++)
        {
            m_dst_data[i][j] = (uint8_t*)malloc(output_frame_sizes[i]);
            REQUIRE_ACTION(NULL != m_dst_data[i], status = HAILO_OUT_OF_HOST_MEMORY, l_clear_buffers, "Out of memory");
        }
    }

    status = HAILO_SUCCESS;
    //TODO remove goto
    goto l_exit;

l_clear_buffers:
    for (size_t i = 0; i < output_vstream_size; i++)
    {
        for (uint8_t j = 0; j < MAX_BATCH; j++)
        {
            FREE(m_dst_data[i][j]);
        }
    }
    (void)hailo_release_output_vstreams(output_vstreams, output_vstream_size);
l_release_input_vstream:
    (void)hailo_release_input_vstreams(input_vstreams, input_vstream_size);
l_exit:
    return status;
}


hailo_status write_all(hailo_input_vstream input_vstream, std::string video_path)
{
    hailo_status status = HAILO_UNINITIALIZED;
    size_t input_frame_size = 0;
    uint32_t i = 0; 
    hailo_input_vstream vstream = input_vstream;
    cv::Mat org_frame;
    cv::Mat pp_frame(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3);  
    cv::Mat input_image;

    std::vector <cv::String > file_names;

    cv::VideoCapture cap(video_path);

    status = hailo_get_input_vstream_frame_size(vstream, &input_frame_size);

    // Prepare src data here
    for (i = 0; i < g_TotalFrame; i++) {
        cap>>org_frame; 
        if (org_frame.empty())
            break;
        cv::resize(org_frame, g_frame[i], cv::Size(FRAME_HEIGHT, FRAME_WIDTH), 1);

        status = hailo_vstream_write_raw_buffer(vstream, g_frame[i].data, input_frame_size);

    }
    g_TotalFrame = i;
    cap.release();
   
    return status;
}

 

hailo_status read_all(hailo_output_vstream output_vstream, std::shared_ptr<FeatureData> feature)
{
    for (size_t i = 0; i < g_TotalFrame; i++) {
        auto &buffer = feature->m_buffers.get_write_buffer();
        hailo_status status = hailo_vstream_read_raw_buffer(output_vstream, buffer.data(), buffer.size());
        feature->m_buffers.release_write_buffer();

        if (HAILO_SUCCESS != status) {
            std::cerr << "Failed reading with status = " <<  status << std::endl;
            return status;
        }
    }

    return HAILO_SUCCESS;
}

hailo_status post_processing_all(std::vector<std::shared_ptr<FeatureData>> &features)
{
    auto status = HAILO_SUCCESS;

    std::sort(features.begin(), features.end(), &FeatureData::sort_tensors_by_size);
    cv::VideoWriter video(PROCESSED_VID_FILE, cv::VideoWriter::fourcc('m','p','4','v'), 30, cv::Size(FRAME_HEIGHT,FRAME_WIDTH));

    for (uint32_t i = 0; i < g_TotalFrame; i++) {
        auto detections = post_processing(
            features[0]->m_buffers.get_read_buffer().data(), features[0]->m_qp_zp, features[0]->m_qp_scale,
            features[1]->m_buffers.get_read_buffer().data(), features[1]->m_qp_zp, features[1]->m_qp_scale,
            features[2]->m_buffers.get_read_buffer().data(), features[2]->m_qp_zp, features[2]->m_qp_scale);
    
        for (auto &feature : features) {
            feature->m_buffers.release_read_buffer();
        }

        for (auto &detection : detections) {
            if (detection.confidence==0) {
                continue;
            }
        
	        {
                cv::rectangle(g_frame[i], cv::Point2f(detection.xmin, detection.ymin), 
                                          cv::Point2f(detection.xmax, detection.ymax), 
                                          cv::Scalar(0, 0, 255), 1);
            }
            std::cout << "Detection: " << get_coco_name_from_int(static_cast<int>(detection.class_id)) << ", Confidence: " << detection.confidence << std::endl;
        }

        video.write(g_frame[i]);

    }
    video.release();

    return status;
}

hailo_status run_network(hailo_configured_network_group network_group, hailo_input_vstream input_vstream, 
                        size_t input_frame_size, hailo_output_vstream *output_vstreams, 
                        size_t output_vstreams_size, std::string video_path)
{
    hailo_status status = HAILO_SUCCESS; // Success oriented
    hailo_activated_network_group activated_network_group = NULL;   
    
    std::cout << "-I- Running network. Input frame size: " << input_frame_size << std::endl;

    cv::VideoCapture cap(video_path);
    g_frame = new cv::Mat[cap.get(cv::CAP_PROP_FRAME_COUNT)];

    status = hailo_activate_network_group(network_group, NULL, &activated_network_group);

    std::vector<std::shared_ptr<FeatureData>> features;
    features.reserve(output_vstreams_size);
    for (size_t i = 0; i < output_vstreams_size; i++) {
        std::shared_ptr<FeatureData> feature(nullptr);
        auto m_status = create_feature(output_vstreams[i], feature);
        if (HAILO_SUCCESS != status) {
            std::cerr << "Failed creating feature with status = " << status << std::endl;
            return m_status;
        }

        features.emplace_back(feature);
    }

    // Create write thread
    auto input_thread(std::async(write_all, input_vstream, video_path));

    // Create read threads
    std::vector<std::future<hailo_status>> output_threads;
    output_threads.reserve(output_vstreams_size);
    for (size_t i = 0; i < output_vstreams_size; i++) {
        output_threads.emplace_back(std::async(read_all, output_vstreams[i], features[i]));
    }

    auto pp_thread(std::async(post_processing_all, std::ref(features)));

    // End threads
    for (size_t i = 0; i < output_threads.size(); i++) {
         status = output_threads[i].get();
    }
    auto input_status = input_thread.get();
    auto pp_status = pp_thread.get();

    if (HAILO_SUCCESS != input_status) {
        std::cerr << "Write thread failed with status " << input_status << std::endl;
        return input_status; 
    }
    if (HAILO_SUCCESS != status) {
        std::cerr << "Read failed with status " << status << std::endl;
        return status;
    }
    if (HAILO_SUCCESS != pp_status) {
        std::cerr << "Post-processing failed with status " << pp_status << std::endl;
        return pp_status;
    }

    status = hailo_deactivate_network_group(activated_network_group);
    return status;
}

std::string get_cmd_option(int argc, char *argv[], const std::string &option)
{
    std::string cmd;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (0 == arg.find(option, 0))
        {
            std::size_t found = arg.find("=", 0) + 1;
            cmd = (arg.substr(found)).data();
            return cmd;
        }
    }
    return cmd;
}


int main(int argc, char** argv) {
    std::string hef_path = get_cmd_option(argc, argv, "-hef=");
    std::string video_path = get_cmd_option(argc, argv, "-video=");
    
    std::cout<<hef_path<<std::endl;
    
    hailo_status status = HAILO_UNINITIALIZED;
    hailo_hef hef = {NULL};
    hailo_configure_params_t configure_params = {};
    hailo_configured_network_group network_group = {NULL};
    hailo_device device = NULL;
    size_t network_groups_size = 1;
    hailo_input_vstream input_vstreams[MAX_EDGE_LAYERS];
    hailo_output_vstream output_vstreams[MAX_EDGE_LAYERS];
    size_t input_frame_size[MAX_EDGE_LAYERS];
    size_t output_frame_size[MAX_EDGE_LAYERS];
    size_t num_output_vstreams = {0};

    size_t number_of_devices = 0;
    hailo_pcie_device_info_t pcie_device_info[8];
    status = hailo_scan_pcie_devices(pcie_device_info, 8, &number_of_devices);
    status = hailo_create_pcie_device(&pcie_device_info[0], &device);
    REQUIRE_SUCCESS(status, l_exit, "Failed to create pcie_device");

    status = hailo_create_hef_file(&hef, hef_path.c_str());
    REQUIRE_SUCCESS(status, l_release_hef, "Failed creating hef file %s", hef_path.c_str());

    status = hailo_init_configure_params(hef, HAILO_STREAM_INTERFACE_PCIE, &configure_params);
    REQUIRE_SUCCESS(status, l_release_hef, "Failed init configure params");

    status = hailo_configure_device(device, hef, &configure_params, &network_group, &network_groups_size);
    REQUIRE_SUCCESS(status, l_release_hef, "Failed configuring devcie");
    REQUIRE_ACTION(network_groups_size == 1, status = HAILO_INVALID_ARGUMENT, l_release_hef, 
        "Unexpected network group size");

    status = build_streams(network_group, 
        input_vstreams, input_frame_size,
        output_vstreams, output_frame_size,
        dst_data, &num_output_vstreams);
    REQUIRE_SUCCESS(status, l_release_hef, "Failed building streams");

    status = run_network(network_group, input_vstreams[0], input_frame_size[0], output_vstreams, num_output_vstreams, video_path);

    delete[] g_frame;

    printf(BOLDBLUE);
    printf("\nInference ran successfully!\n\n");
    printf(RESET);

    status = HAILO_SUCCESS;

    l_release_hef:
        if (NULL != hef)
        {
            (void)hailo_release_hef(hef);            
        }
        (void) hailo_release_device(device);
    l_exit:
        return status;
}