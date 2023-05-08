/**
 * Copyright (c) 2020-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
/**
 * @file vstreams_example
 * This example demonstrates using virtual streams over c++
 **/

#include "include/infer.hpp"

#include "hailo/hailort.hpp"

#include <iostream>
#include <thread>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include<stdio.h>

constexpr int FEATURE_MAP_SIZE1 = 20;
constexpr int FEATURE_MAP_SIZE2 = 40;
constexpr int FEATURE_MAP_SIZE3 = 80;
constexpr int FEATURE_MAP_CHANNELS = 85;
constexpr int ANCHORS_NUM = 3;
constexpr int FLOAT = 4;

using namespace hailort;

Expected<std::shared_ptr<ConfiguredNetworkGroup>> configure_network_group(VDevice &vdevice, const char* hef_path)
{
    auto hef = Hef::create(hef_path);
    if (!hef) {
        return make_unexpected(hef.status());
    }

    auto configure_params = vdevice.create_configure_params(hef.value());
    if (!configure_params) {
        return make_unexpected(configure_params.status());
    }

    auto network_groups = vdevice.configure(hef.value(), configure_params.value());
    if (!network_groups) {
        return make_unexpected(network_groups.status());
    }

    if (1 != network_groups->size()) {
        std::cerr << "Invalid amount of network groups" << std::endl;
        return make_unexpected(HAILO_INTERNAL_FAILURE);
    }

    return std::move(network_groups->at(0));
}

inline bool ends_with(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

void write_all(InputVStream &input, const char* images_path, hailo_status &status)
{
    std::vector<cv::String> file_names;
    cv::glob(images_path, file_names, false);
    assert(("currently support only 1 frame. For more frames, change c# code to send bigger buffer, and use pointer arithmetics in c++ code.", file_names.size() <= 1));
    
    for (std::string file : file_names) {
        if (not(ends_with(file, ".jpg") || ends_with(file, ".png") || ends_with(file, ".jpeg")))
            continue;
        auto rgb_frame = cv::imread(file,  cv::IMREAD_COLOR);
        
        if (rgb_frame.channels() == 3)
            cv::cvtColor(rgb_frame, rgb_frame, cv::COLOR_BGR2RGB);
    
        
        if (rgb_frame.rows != input.get_info().shape.height || rgb_frame.cols != input.get_info().shape.width)
            cv::resize(rgb_frame, rgb_frame, cv::Size(input.get_info().shape.width, input.get_info().shape.height), cv::INTER_AREA);
        
        auto status = input.write(MemoryView(rgb_frame.data, input.get_frame_size()));
            if (HAILO_SUCCESS != status) 
                return;
    }

    // // Flushing is not mandatory here
    // status = input.flush();
    // if (HAILO_SUCCESS != status) {
    //     std::cerr << "Failed flushing input vstream" << std::endl;
    //     return;
    // }

    status = HAILO_SUCCESS;
    return;
}

void read_all(OutputVStream &output, const char* images_path, hailo_status &status, float32_t* arr, size_t arr_size)
{
    // std::vector<float32_t> data(output.get_frame_size()); // output.get_user_buffer_format().type HAILO_FORMAT_TYPE_UINT16 // TODO: remove uint16_t
    std::vector<cv::String> file_names;
    cv::glob(images_path, file_names, false);
    size_t num_frames = 0;
    for (std::string file : file_names) {
        if (not(ends_with(file, ".jpg") || ends_with(file, ".png") || ends_with(file, ".jpeg")))
            continue;
        // arr[num_frames] because batch-size may be more than one image, and we still need to return a full tensor Nchw or whatever another order
        auto status = output.read(MemoryView(arr, arr_size)); // arr + num_frames // pointer arithmetic // auto status = output.read(MemoryView(arr, arr_size));
        if (HAILO_SUCCESS != status) {
            std::cout << "failed with status " << status << std::endl;
            return;
        }
        num_frames++;
        // auto detected_class = classification_post_process<T>(data);
        // std::cout << "-I- [" << num_frames << "] Detected class: " << detected_class << std::endl;
    }
    status = HAILO_SUCCESS;
    return;
}

hailo_status infer(std::vector<InputVStream> &input_streams, std::vector<OutputVStream> &output_streams, const char* images_path, std::vector<std::pair<float32_t*, size_t>> out_tensors)
{

    hailo_status status = HAILO_SUCCESS; // Success oriented
    hailo_status input_status[input_streams.size()] = {HAILO_UNINITIALIZED};
    hailo_status output_status[output_streams.size()] = {HAILO_UNINITIALIZED};
    std::unique_ptr<std::thread> input_threads[input_streams.size()];
    std::unique_ptr<std::thread> output_threads[output_streams.size()];
    size_t input_thread_index = 0;
    size_t output_thread_index = 0;

    // Create write threads
    for (input_thread_index = 0 ; input_thread_index < input_streams.size(); input_thread_index++) {
        input_threads[input_thread_index] = std::make_unique<std::thread>(write_all,
            std::ref(input_streams[input_thread_index]), images_path, std::ref(input_status[input_thread_index]));
    }

    // Create read threads
    for (output_thread_index = 0 ; output_thread_index < output_streams.size(); output_thread_index++) {
        output_threads[output_thread_index] = std::make_unique<std::thread>(read_all,
            std::ref(output_streams[output_thread_index]), images_path, 
            std::ref(output_status[output_thread_index]), out_tensors[output_thread_index].first, out_tensors[output_thread_index].second);
    }

    // Join write threads
    for (size_t i = 0; i < input_thread_index; i++) {
        input_threads[i]->join();
        if (HAILO_SUCCESS != input_status[i]) {
            status = input_status[i];
            std::cout << "failed in thread write " << i << std::endl;
        }
    }

    // Join read threads
    for (size_t i = 0; i < output_thread_index; i++) {
        output_threads[i]->join();
        if (HAILO_SUCCESS != output_status[i]) {
            status = output_status[i];
            std::cout << "failed in thread read " << i << std::endl;
        }
    }

    if (HAILO_SUCCESS == status) {
        std::cout << "Inference finished successfully" << std::endl;
    }

    return status;
}

extern "C" int infer_wrapper(const char* hef_path, const char* images_path, 
    float* arr1, size_t n1,
    float* arr2, size_t n2,
    float* arr3, size_t n3)
{
    std::cout << "successfully loaded libinfer.so" << std::endl;
    std::cout << "hef path entered: " << std::string(hef_path) << std::endl;

    auto vdevice = VDevice::create();
    if (!vdevice) {
        std::cerr << "Failed create vdevice, status = " << vdevice.status() << std::endl;
        return vdevice.status();
    }

    auto network_group = configure_network_group(*vdevice.value(), hef_path);
    if (!network_group) {
        std::cerr << "Failed to configure network group " << hef_path << std::endl;
        return network_group.status();
    }

    auto input_vstream_params = network_group.value()->make_input_vstream_params(false, HAILO_FORMAT_TYPE_UINT8, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    auto output_vstream_params = network_group.value()->make_output_vstream_params(false, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE); // HAILO_FORMAT_TYPE_UINT16
    auto input_vstreams  = VStreamsBuilder::create_input_vstreams(*network_group.value(), input_vstream_params.value());
    auto output_vstreams = VStreamsBuilder::create_output_vstreams(*network_group.value(), output_vstream_params.value());
    if (!input_vstreams or !output_vstreams) {
        std::cerr << "-E- Failed creating input: " << input_vstreams.status() << " output status:" << output_vstreams.status() << std::endl;
        return input_vstreams.status();
    }
    auto vstreams = std::make_pair(input_vstreams.release(), output_vstreams.release());

    std::vector<std::pair<float32_t*, size_t>> out_tensors;
    out_tensors.push_back(std::pair<float32_t*, size_t>(arr3, n3));
    out_tensors.push_back(std::pair<float32_t*, size_t>(arr2, n2));
    out_tensors.push_back(std::pair<float32_t*, size_t>(arr1, n1));

    auto status = infer(vstreams.first, vstreams.second, images_path, out_tensors);
    if (HAILO_SUCCESS != status) {
        std::cerr << "Inference failed "  << status << std::endl;
        return status;
    }

    return HAILO_SUCCESS;
}

int main() {
    size_t n1 = FEATURE_MAP_SIZE1 * FEATURE_MAP_SIZE1 * FEATURE_MAP_CHANNELS * ANCHORS_NUM * FLOAT; // ? * float??
    std::vector<float32_t> arr1(n1);
    size_t n2 = FEATURE_MAP_SIZE2 * FEATURE_MAP_SIZE2 * FEATURE_MAP_CHANNELS * ANCHORS_NUM * FLOAT;
    std::vector<float32_t> arr2(n2);
    size_t n3 = FEATURE_MAP_SIZE3 * FEATURE_MAP_SIZE3 * FEATURE_MAP_CHANNELS * ANCHORS_NUM * FLOAT;
    std::vector<float32_t> arr3(n3);

    infer_wrapper("/home/batshevak/projects/new_jj/Hailo-Application-Code-Examples/infer_wrapper/infer_wrapper/yolov5m_wo_spp_60p.hef", 
    "/home/batshevak/projects/new_jj/Hailo-Application-Code-Examples/infer_wrapper/infer_wrapper/images", &arr1[0], n1, &arr2[0], n2, &arr3[0], n3);
    
}