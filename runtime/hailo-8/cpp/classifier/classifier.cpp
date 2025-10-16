/**
 * Copyright 2021 (C) Hailo Technologies Ltd.
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
 * @ file classifier_example
 * This example demonstrates using virtual streams over c++
 **/

#include "hailo/hailort.hpp"

#include <iostream>
#include <thread>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "imagenet_labels.hpp"

constexpr int WIDTH  = 224;
constexpr int HEIGHT = 224;

using hailort::Device;
using hailort::Hef;
using hailort::Expected;
using hailort::make_unexpected;
using hailort::ConfiguredNetworkGroup;
using hailort::VStreamsBuilder;
using hailort::InputVStream;
using hailort::OutputVStream;
using hailort::MemoryView;

// http://www.jclay.host/dev-journal/simple_cpp_argmax_argmin.html
template <typename T, typename A>
int argmax(std::vector<T, A> const& vec) {
    return static_cast<int>(std::distance(vec.begin(), max_element(vec.begin(), vec.end())));
}

template <typename T, typename A>
std::vector<T, A> softmax(std::vector<T, A> const& vec) {
    std::vector<T, A> result;
    float m = -INFINITY;
    float sum = 0.0;

    for (const auto &val : vec) m = (val>m) ? val : m;
    for (const auto &val : vec) sum += expf(val - m);
    for (const auto &val : vec) result.push_back(expf(val-m)/sum);
    
    return result;   
}

std::string getCmdOption(int argc, char *argv[], const std::string &option)
{
    std::string cmd;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (0 == arg.find(option, 0))
        {
            std::size_t found = arg.find("=", 0) + 1;
            cmd = arg.substr(found, 200);
            return cmd;
        }
    }
    return cmd;
}

Expected<std::shared_ptr<ConfiguredNetworkGroup>> configure_network_group(Device &device, const std::string &hef_file)
{
    auto hef = Hef::create(hef_file);
    if (!hef) {
        return make_unexpected(hef.status());
    }

    auto configure_params = hef->create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
    if (!configure_params) {
        return make_unexpected(configure_params.status());
    }

    auto network_groups = device.configure(hef.value(), configure_params.value());
    if (!network_groups) {
        return make_unexpected(network_groups.status());
    }

    if (1 != network_groups->size()) {
        std::cerr << "Invalid amount of network groups" << std::endl;
        return make_unexpected(HAILO_INTERNAL_FAILURE);
    }

    return std::move(network_groups->at(0));
}

template <typename T=InputVStream>
std::string info_to_str(T &stream)
{
    std::string result = stream.get_info().name;
    result += " (";
    result += std::to_string(stream.get_info().shape.height);
    result += ", ";
    result += std::to_string(stream.get_info().shape.width);
    result += ", ";
    result += std::to_string(stream.get_info().shape.features);
    result += ")";
    return result;
}

template <typename T>
hailo_status write_all(std::vector<InputVStream> &input, std::string &video_path)
{
    std::vector<cv::String> file_names;
    cv::glob(video_path, file_names, false);
    std::cout << "-I- Started write thread " << video_path << std::endl;
    
    for (std::string file : file_names) {
        if (not(file.ends_with(".jpg") || file.ends_with(".png")))
            continue;
        auto rgb_frame = cv::imread(file,  cv::IMREAD_COLOR);
        
        if (rgb_frame.channels() == 3)
            cv::cvtColor(rgb_frame, rgb_frame, cv::COLOR_BGR2RGB);
    
        
        if (rgb_frame.rows != HEIGHT || rgb_frame.cols != WIDTH)
            cv::resize(rgb_frame, rgb_frame, cv::Size(WIDTH, HEIGHT), cv::INTER_AREA);
        
        int factor = std::is_same<T, uint8_t>::value ? 1 : 4;                                  // In case we use float32_t, we have 4 bytes per component
        auto status = input[0].write(MemoryView(rgb_frame.data, HEIGHT * WIDTH * 3 * factor)); // Writing HEIGHT * WIDTH, 3 channels of uint8
            if (HAILO_SUCCESS != status) 
                return status;
    }
    return HAILO_SUCCESS;
}

template <typename T>
std::string classification_post_process(std::vector<T>& logits, bool do_softmax=false, float threshold=0.3) 
{
    int max_idx;
    static ImageNetLabels obj;
    std::vector<T> softmax_result(logits);
    if (do_softmax) {
	softmax_result = softmax(logits);
        max_idx = argmax(softmax_result);
    } else 
        max_idx = argmax(logits);
    if (softmax_result[max_idx] < threshold) return "N\\A";
    return obj.imagenet_labelstring(max_idx) + " (" + std::to_string(softmax_result[max_idx]) + ")";
}

template <typename T>
hailo_status read_all(OutputVStream &output, std::string &video_path)
{
    std::vector<T> data(output.get_frame_size());
    std::vector<cv::String> file_names;
    std::cout << "-I- Started read thread " << std::endl;
    cv::glob(video_path, file_names, false);
    size_t num_frames = 0;
    for (std::string file : file_names) {
        if (not(file.ends_with(".jpg") || file.ends_with(".png")))
            continue;
        
        auto status = output.read(MemoryView(data.data(), data.size()));
        if (HAILO_SUCCESS != status)
            return status;
        num_frames++;
        auto detected_class = classification_post_process<T>(data);
        std::cout << "-I- [" << num_frames << "] Detected class: " << detected_class << std::endl;
    }
    std::cout << "-I- Finished read thread " << std::endl;
    return HAILO_SUCCESS;
}

void print_net_banner(std::pair< std::vector<InputVStream>, std::vector<OutputVStream> > &vstreams) {
    std::cout << "-I---------------------------------------------------------------------" << std::endl;
    std::cout << "-I- Dir  Name                                     " << std::endl;
    std::cout << "-I---------------------------------------------------------------------" << std::endl;
    for (auto &value: vstreams.first)
        std::cout << "-I- IN:  " << info_to_str<InputVStream>(value) << std::endl;
    std::cout << "-I---------------------------------------------------------------------" << std::endl;
    for (auto &value: vstreams.second)
        std::cout << "-I- OUT: " << info_to_str<OutputVStream>(value) << std::endl;
    std::cout << "-I---------------------------------------------------------------------" << std::endl;
}

template <typename IN_T, typename OUT_T>
hailo_status infer(std::vector<InputVStream> &inputs, std::vector<OutputVStream> &outputs, std::string video_path)
{
    hailo_status input_status = HAILO_UNINITIALIZED;
    hailo_status output_status = HAILO_UNINITIALIZED;
    std::vector<std::thread> output_threads;

    std::thread input_thread([&inputs, &video_path, &input_status]() { input_status = write_all<IN_T>(inputs, video_path); });
    
    for (auto &output: outputs)
        output_threads.push_back( std::thread([&output, &video_path, &output_status]() { output_status = read_all<OUT_T>(output, video_path); }) );

    input_thread.join();
    
    for (auto &out: output_threads)
        out.join();

    if ((HAILO_SUCCESS != input_status) || (HAILO_SUCCESS != output_status)) {
        return HAILO_INTERNAL_FAILURE;
    }

    std::cout << "-I- Inference finished successfully" << std::endl;
    return HAILO_SUCCESS;
}

int main(int argc, char**argv)
{
    std::string hef_file   = getCmdOption(argc, argv, "-hef=");
    std::string video_path = getCmdOption(argc, argv, "-path=");
    auto all_devices       = Device::scan_pcie();
    std::cout << "-I- images path: " << video_path << std::endl;
    std::cout << "-I- hef: " << hef_file << std::endl;

    auto device = Device::create_pcie(all_devices.value()[0]);
    if (!device) {
        std::cerr << "-E- Failed create_pcie " << device.status() << std::endl;
        return device.status();
    }

    auto network_group = configure_network_group(*device.value(), hef_file);
    if (!network_group) {
        std::cerr << "-E- Failed to configure network group " << hef_file << std::endl;
        return network_group.status();
    }
    
    auto input_vstream_params = network_group.value()->make_input_vstream_params(true, HAILO_FORMAT_TYPE_UINT8, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    auto output_vstream_params = network_group.value()->make_output_vstream_params(false, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    auto input_vstreams  = VStreamsBuilder::create_input_vstreams(*network_group.value(), input_vstream_params.value());
    auto output_vstreams = VStreamsBuilder::create_output_vstreams(*network_group.value(), output_vstream_params.value());
    if (!input_vstreams or !output_vstreams) {
        std::cerr << "-E- Failed creating input: " << input_vstreams.status() << " output status:" << output_vstreams.status() << std::endl;
        return input_vstreams.status();
    }
    auto vstreams = std::make_pair(input_vstreams.release(), output_vstreams.release());

    print_net_banner(vstreams);

    auto activated_network_group = network_group.value()->activate();
    if (!activated_network_group) {
        std::cerr << "-E- Failed activated network group " << activated_network_group.status();
        return activated_network_group.status();
    }
    
    auto status  = infer<uint8_t, float32_t>(vstreams.first, vstreams.second, video_path);

    if (HAILO_SUCCESS != status) {
        std::cerr << "-E- Inference failed "  << status << std::endl;
        return status;
    }

    return HAILO_SUCCESS;
}
