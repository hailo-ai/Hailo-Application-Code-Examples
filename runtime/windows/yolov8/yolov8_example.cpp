/**
 * Copyright 2024 (C) Hailo Technologies Ltd.
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
 * @file yolov8_example
 **/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <future>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include "opencv2/core/types.hpp"
#include "opencv2/core/bufferpool.hpp"
#include <type_traits>
#include "opencv2/core/mat.inl.hpp"

#include "common.h"
#include "hailo/hailort.h"

#define MAX_EDGE_LAYERS (16)
#define MAX_BATCH (64)
#define FRAME_WIDTH (640)
#define FRAME_HEIGHT (640)

uint8_t *dst_data[MAX_EDGE_LAYERS][MAX_BATCH] = {NULL};
extern unsigned int box_index;

// Define a struct to hold frame and detection information
struct FrameData {
    cv::Mat frame;
};

std::queue<FrameData> frameQueue;
std::mutex queueMutex;

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
    hailo_input_vstream vstream = input_vstream;
    cv::Mat org_frame;

    std::vector <cv::String > file_names;

    cv::VideoCapture cap(std::stoi(video_path));

    status = hailo_get_input_vstream_frame_size(vstream, &input_frame_size);

    cv::Size target_size(FRAME_HEIGHT, FRAME_WIDTH);
    while (true) {
        cap >> org_frame; 
      
        if (org_frame.empty()) {
            std::cerr << "Error: Unable to capture frame." << std::endl;
            break;
        }

         // Add the frame to the queue
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            frameQueue.push({org_frame});
        }

         cv::resize(org_frame, org_frame, target_size, 1);

        status = hailo_vstream_write_raw_buffer(vstream, org_frame.data, input_frame_size);
    
    }

    cap.release();
   
    return status;
}
 

hailo_status read_all(hailo_output_vstream output_vstream, std::vector<float32_t>& data)
{
    while (true) {
        hailo_status status = hailo_vstream_read_raw_buffer(output_vstream, data.data() , data.size());

        if (HAILO_SUCCESS != status) {
            std::cerr << "Failed reading with status = " <<  status << std::endl;
            return status;
        }
    }

    return HAILO_SUCCESS;
}

void print_boxes_coord_per_class(std::vector<float32_t> data, cv::Mat& frame, float32_t thr=0.35)
{
    size_t index=-1;
    size_t class_idx=0;
    while (class_idx<80) {
        auto num_of_class_boxes = data.at(++index);
        for (auto box_idx=0;box_idx<num_of_class_boxes;box_idx++) {
            auto y_min = data.at(++index)*480;
            auto x_min = data.at(++index)*FRAME_WIDTH;
            auto y_max = data.at(++index)*480;
            auto x_max = data.at(++index)*FRAME_WIDTH;
            auto confidence = data.at(++index);
            
            if (confidence>=thr) {
                cv::rectangle(frame, cv::Point2f(x_min, y_min), 
                                    cv::Point2f(x_max, y_max), 
                                    cv::Scalar(0, 0, 255), 1);

                cv::putText(frame, get_coco_name_from_int(static_cast<int>(class_idx + 1)),
                        cv::Point(x_min, y_min - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
       
                std::string label = get_coco_name_from_int(class_idx + 1);
                std::cout << "-I- Class [" << label << "] box ["<< box_idx << "] conf: " << confidence << ": ";
                std::cout << "(" << x_min << ", " << y_min << ", " << x_max << ", " << y_max << ")" << std::endl;
            }
            //cv::imshow("Postprocessing Video",frame);
           // cv::waitKey(10);
        }
        class_idx++;
    }
    cv::imshow("Postprocessing Video",frame);
    cv::waitKey(10);
}

hailo_status post_processing_all(std::vector<float32_t>& data)
{
    auto status = HAILO_SUCCESS;
    FrameData frameData;
    cv::Mat frame;

    while(true)
    {
        // Pop a frame from the queue
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            if (!frameQueue.empty()) {
                frameData = frameQueue.front();
                frameQueue.pop();
            }
        }

        if (frameData.frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(30)); // Sleep for a short time if the queue is empty
            continue;
        }

        print_boxes_coord_per_class(data, std::ref(frameData.frame), 0.50);
        

    }
    return status;
}


hailo_status run_network(hailo_configured_network_group network_group, hailo_input_vstream input_vstream, 
                        size_t input_frame_size, hailo_output_vstream *output_vstreams, 
                        size_t output_vstreams_size, std::string video_path)
{
    hailo_status status = HAILO_SUCCESS; // Success oriented
    hailo_activated_network_group activated_network_group = NULL;   
    
    std::cout << "-I- Running network. Input frame size: " << input_frame_size << std::endl;

    cv::VideoCapture cap(std::stoi(video_path));

    //botner
    if (!cap.isOpened()) {
        std::cerr << "Error: Couldn't open the video file or camera." << std::endl;
        return HAILO_INVALID_OPERATION;
    }
  
    status = hailo_activate_network_group(network_group, NULL, &activated_network_group);

     // Create write thread
    auto input_thread(std::async(&write_all, input_vstream, video_path));
    std::vector<float32_t> vstream_output_data(160320);

    // Create read threads
    std::future<hailo_status> output_thread = std::async(&read_all, output_vstreams[0], std::ref(vstream_output_data));
 
    std::future<hailo_status> pp_thread = std::async(&post_processing_all, std::ref(vstream_output_data));

    status = output_thread.get();

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