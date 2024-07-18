/**
 * Copyright (c) 2020-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
/**
 * @file dsp_utils test program
 * This example demonstrates using image resizing with dsp_utils
 **/

#include <chrono>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include "media_library/dsp_utils.hpp"
#define CROP_CHECK
int main()
{
    std::chrono::system_clock::time_point start, end;

    cv::Mat input_cvimage = cv::imread("zidane_FHD.jpg");
    if (input_cvimage.empty() == true) {
        std::cout << "[ERROR] : cv::imread()" << std::endl;
        exit(-1);
    }

    dsp_status status;

    // Acquire DSP device

    status = dsp_utils::acquire_device();
    if (DSP_SUCCESS != status) {
        std::cout << "[ERROR] : acquire_device()" << std::endl;
        exit(-1);
    }

    // Create input buffer

    constexpr size_t input_width = 1920;
    constexpr size_t input_height = 1080;

    constexpr size_t input_bytesperline = input_width * 3;

    auto input_size = input_width * input_height * 3;
    void *input_plane = NULL;
 
    auto dsp_status = dsp_utils::create_hailo_dsp_buffer(input_size, &input_plane);

    if (DSP_SUCCESS != dsp_status) {
        std::cout << "[ERROR] : create_hailo_dsp_buffer(input_plane)" << std::endl;
        exit(-1);
    }

    if (input_cvimage.isContinuous()) {
        std::memcpy(input_plane, input_cvimage.data, input_size);
    } 
    else {
        for (int i = 0; i < input_cvimage.rows; ++i) {
            std::memcpy((char*)input_plane + i * input_cvimage.cols * input_cvimage.channels(), input_cvimage.ptr(i), input_cvimage.cols * input_cvimage.channels());
        }
    }
 
    std::vector<dsp_data_plane_t> input_planes;
    auto input_rgb_plane = (dsp_data_plane_t) {
        .userptr = input_plane,
        .bytesperline = input_bytesperline,
        .bytesused = input_size,
    };

    input_planes.push_back(input_rgb_plane);

    auto input_image = (dsp_image_properties_t) {
        .width = input_width,
        .height = input_height,
        .planes = (dsp_data_plane_t*)input_planes.data(),
        .planes_count = 1,
        .format = DSP_IMAGE_FORMAT_RGB
    };

    // Create output buffer

#ifndef CROP_CHECK
    constexpr size_t output_width = 320;
    constexpr size_t output_height = 240;
#else
    constexpr size_t output_width = 120;
    constexpr size_t output_height = 120;
#endif

    constexpr size_t output_bytesperline = output_width * 3;

    auto output_size = output_width * output_height * 3;
    void *output_plane = NULL;
    dsp_status = dsp_utils::create_hailo_dsp_buffer(output_size, &output_plane);

    if (DSP_SUCCESS != dsp_status) {
        std::cout << "[ERROR] : create_hailo_dsp_buffer(output_plane)" << std::endl;
        exit(-1);
    }

    cv::Mat output_cvimage = cv::Mat(output_height, output_width, CV_8UC3, output_plane); 

    std::vector<dsp_data_plane_t> output_planes;
    auto output_rgb_plane = (dsp_data_plane_t) {
        .userptr = output_plane,
        .bytesperline = output_bytesperline,
        .bytesused = output_size,
    };
    output_planes.push_back(output_rgb_plane);

    auto output_image = (dsp_image_properties_t) {
        .width = output_width,
        .height = output_height,
        .planes = (dsp_data_plane_t*)output_planes.data(),
        .planes_count = 1,
        .format = DSP_IMAGE_FORMAT_RGB,
    };

    // Perform crop and resize

    auto args = (dsp_utils::crop_resize_dims_t) {
#ifndef CROP_CHECK
        .perform_crop = 0,
#else
        .perform_crop = 1,
#endif
         .crop_start_x = 1375,
        .crop_end_x = 1608,
        .crop_start_y = 162,
        .crop_end_y = 395,
        .destination_width = output_width,
        .destination_height = output_height
    };

    start = std::chrono::system_clock::now();
    status = dsp_utils::perform_crop_and_resize(&input_image, &output_image, args, INTERPOLATION_TYPE_NEAREST_NEIGHBOR);
    end = std::chrono::system_clock::now();
    if (DSP_SUCCESS != status) {
        std::cout << "[ERROR] : dsp_utils::perform_crop_and_resize()" << std::endl;
        exit(-1);
    }

    double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "dsp_utils::perform_crop_and_resize() : " << elapsed / 1000.0f << " (ms)" << std::endl;

     // Release DSP device

    status = dsp_utils::release_device();
    if (DSP_SUCCESS != status) {
        std::cout << "[ERROR] : dsp_release_device()" << std::endl;
        exit(-1);
    }

    cv::imwrite("output_dsp.jpg", output_cvimage);

    // Perform resize with OpenCV

    start = std::chrono::system_clock::now();
#ifdef CROP_CHECK
    input_cvimage = cv::Mat(input_cvimage, cv::Rect(1375, 162, 233, 233));
#endif
    cv::resize(input_cvimage, output_cvimage, cv::Size(output_width, output_height), cv::INTER_NEAREST);
    end = std::chrono::system_clock::now();

    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "cv::resize() : " << elapsed / 1000.0f << " (ms)" << std::endl;

    cv::imwrite("output_cpu.jpg", output_cvimage);

    return 0;
}
