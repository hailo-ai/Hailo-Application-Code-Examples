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
 * @ file hailo_ort_example
 * This example demonstrates using virtual streams over c++
 **/

#include "hailo/hailort.hpp"

#include <cxxabi.h>
#include <iostream>
#include <chrono>
#include <mutex>
#include <thread>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp> 

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <fstream>
#include <array>
#include <typeinfo>
#include <iomanip>

#include "utils.h"

constexpr bool QUANTIZED = false;
constexpr hailo_format_type_t FORMAT_TYPE = HAILO_FORMAT_TYPE_AUTO;
constexpr int CV_TYPE = CV_32FC3;
std::mutex m;
std::mutex m_data;

#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

using hailort::Device;
using hailort::Hef;
using hailort::Expected;
using hailort::make_unexpected;
using hailort::ConfiguredNetworkGroup;
using hailort::VStreamsBuilder;
using hailort::InputVStream;
using hailort::OutputVStream;
using hailort::MemoryView;


void print_inference_statistics(std::size_t num_of_frames, double total_time_hailo, double total_time_onnx) {
    std::cout << BOLDGREEN << "-I-----------------------------------------------" << std::endl;
    std::cout << "-I- Total ONNXRuntime Time:   " << total_time_onnx << " sec" << std::endl;
    std::cout << "-I- Total Hailo Time:         " << total_time_hailo << " sec" << std::endl;
    std::cout << "-I- Total Time:               " << total_time_onnx + total_time_hailo << " sec" << std::endl;
    std::cout << "-I-----------------------------------------------" << std::endl;
    std::cout << "-I- Average ONNXRuntime FPS:  " << (double)num_of_frames /total_time_onnx << std::endl;
    std::cout << "-I- Average Hailo FPS:        " << (double)num_of_frames / total_time_hailo << std::endl;
    std::cout << "-I- Average FPS:              " << (double)num_of_frames / (total_time_onnx + total_time_hailo) << std::endl;
    std::cout << "-I-----------------------------------------------" << std::endl;
    std::cout << "-I- ONNXRuntime Latency:      " << 1.0 / ((double)num_of_frames /total_time_onnx)*1000 << " ms" << std::endl;
    std::cout << "-I- Hailo Latency:            " << 1.0 / ((double)num_of_frames / total_time_hailo)*1000 << " ms"<< std::endl;
    std::cout << "-I- Total Latency:            " << 1.0 / ((double)num_of_frames / (total_time_onnx + total_time_hailo))*1000 << " ms" << std::endl;
    std::cout << "-I-----------------------------------------------\n" << std::endl << RESET;
}


std::string info_to_str(hailo_vstream_info_t vstream_info) {
    std::string result = vstream_info.name;
    result += " (";
    result += std::to_string(vstream_info.shape.height);
    result += ", ";
    result += std::to_string(vstream_info.shape.width);
    result += ", ";
    result += std::to_string(vstream_info.shape.features);
    result += ")";
    return result;
}


std::vector<Ort::Value> onnxruntime_inference(Ort::AllocatorWithDefaultOptions& ort_alloc, 
                                                    Ort::Session& session,
                                                    std::chrono::duration<double>& onnx_elapsed_time_s_vec, 
                                                    std::vector<std::vector<float32_t>> data_vector, std::size_t num_of_frames) {
    size_t input_num = session.GetInputCount();
    size_t output_num = session.GetOutputCount();
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    std::vector<const char*> input_names(input_num);
    std::vector<const char*> output_names(output_num);

    for (size_t i = 0; i < input_num; i++) {
        input_names[i] = session.GetInputNameAllocated(i, ort_alloc).release();
    }

    for (size_t i = 0; i < output_num; i++) {;
        output_names[i] = session.GetOutputNameAllocated(i, ort_alloc).release();
    }

    std::vector<Ort::Value> input_tensors;
    std::vector<std::vector<int64_t>> input_shapes;

    std::vector<cv::Mat> input_images;

    for (size_t i = 0; i < input_num; i++){
        auto input_shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        if (input_shape[0] == -1)
            input_shape[0] = 1;
        cv::Mat image(input_shape[2],input_shape[3],CV_32FC(input_shape[1]));
        input_images.push_back(image);
        input_shapes.push_back(input_shape);
    }

    for (size_t i = 0; i < output_num; i++){
        auto output_shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        if (output_shape[0] == -1)
            output_shape[0] = 1;

        input_shapes.push_back(output_shape);
    }
    std::vector<std::vector<float>> images_vector(input_num);
    for (size_t i = 0; i < images_vector.size(); i++) {
        images_vector[i].assign((float*)input_images[i].data, (float*)input_images[i].data + input_images[i].total()*input_images[i].channels());
    }

    for (size_t i = 0; i < input_num; i++){
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, images_vector[i].data(), images_vector[i].size(),
            input_shapes[i].data(), input_shapes[i].size()
            ));
    }

    std::vector<Ort::Value> output_tensors;

    std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_of_frames; i++) {
        output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_num, output_names.data(), output_num);
        assert(output_tensors[0].IsTensor());
    }
    std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::high_resolution_clock::now();

    onnx_elapsed_time_s_vec = t_end - t_start;

    return output_tensors;
}


template<typename T=float32_t>
hailo_status read_all(OutputVStream &output, Ort::AllocatorWithDefaultOptions& ort_alloc, Ort::Session& session,
                    std::size_t num_of_frames, std::chrono::time_point<std::chrono::system_clock>& start_time,
                    std::chrono::duration<double>& elapsed_time_s, std::vector<std::vector<T>>& data_vector) {
    m.lock();
    std::cout << BOLDCYAN << "-I- Started read thread: " << info_to_str(output.get_info()) << std::endl << RESET;
    m.unlock();
    std::vector<T> data(output.get_frame_size());
    hailo_status status = HAILO_SUCCESS;
    for (size_t i = 0; i < num_of_frames; i++) {
        status = output.read(MemoryView(data.data(), data.size()));
        cv::Mat imageMat(output.get_info().shape.height, output.get_info().shape.width, CV_8U, data.data());
        if (HAILO_SUCCESS != status) {
            std::cerr << "Failed reading with status = " <<  status << std::endl;
            return status;
        }

        std::cout << YELLOW << std::ends;
        printf("\r-I-  Recv %lu/%lu",i+1, num_of_frames);
        std::cout << RESET << std::ends;
    }
    std::chrono::time_point<std::chrono::system_clock> end_t = std::chrono::high_resolution_clock::now();
    elapsed_time_s = end_t - start_time;

    m_data.lock();
    data_vector.push_back(data);
    m_data.unlock();

    if (HAILO_SUCCESS != status) {
        return status;
    }
    return status;
}


template<typename T=float32_t>
hailo_status write_all(InputVStream &input, std::size_t num_of_frames,
                        std::chrono::time_point<std::chrono::system_clock>& start_time, cv::Mat image) {
    m.lock();
    std::cout << BOLDWHITE << "-I- Started write thread: " << info_to_str(input.get_info()) << std::endl << RESET;
    m.unlock();

    std::vector<T> buff(input.get_frame_size());
    
    auto shape = input.get_info().shape;
    
    hailo_status status = HAILO_SUCCESS;
    start_time = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_of_frames; i++) {
        
        int factor = std::is_same<T, uint8_t>::value ? 1 : 4;
        status = input.write(MemoryView(image.data, shape.height * shape.width * shape.features * factor));
    }
    if (HAILO_SUCCESS != status) {
        return status;
    }
    return status;
}


template<typename IT=float32_t, typename OT=float32_t>
hailo_status infer(std::vector<InputVStream> &inputs, std::vector<OutputVStream> &outputs, std::size_t num_of_frames,  
                    std::chrono::time_point<std::chrono::system_clock>& start_time,
                    std::chrono::duration<double>& elapsed_time_s, 
                    std::chrono::duration<double>& onnx_elapsed_time_s, 
                    Ort::AllocatorWithDefaultOptions& ort_alloc, Ort::Session& session, cv::Mat& image) {
    hailo_status input_status = HAILO_UNINITIALIZED;
    hailo_status output_status = HAILO_UNINITIALIZED;
    std::vector<std::thread> input_threads;
    std::vector<std::thread> output_threads;
    std::vector<std::vector<OT>> data_vector;

    for (auto &input: inputs)
        input_threads.push_back( std::thread(
            [&input, &num_of_frames, &start_time, &image, &input_status]() 
            { input_status = write_all<IT>(input, num_of_frames, start_time, image); }
            ) );
    
    for (auto &output: outputs)
        output_threads.push_back( std::thread(
            [&output, &ort_alloc, &session, &num_of_frames, &start_time, &elapsed_time_s, &data_vector, &output_status]() 
            { output_status = read_all<OT>(output, ort_alloc, session, num_of_frames, start_time, elapsed_time_s, std::ref(data_vector)); }
            ) );

    for (auto &in: input_threads)
        in.join();
    
    for (auto &out: output_threads)
        out.join();

    if ((HAILO_SUCCESS != input_status) || (HAILO_SUCCESS != output_status)) {
        return HAILO_INTERNAL_FAILURE;
    }

    auto onnx_output = onnxruntime_inference(ort_alloc, session, onnx_elapsed_time_s, data_vector, num_of_frames);

    std::cout << BOLDBLUE << "\n\n-I- Inference finished successfully\n" << std::endl << RESET;
    return HAILO_SUCCESS;
}


void print_net_banner(std::string onnx_file, std::string hef_file, 
                        std::pair< std::vector<InputVStream>, std::vector<OutputVStream> > &vstreams,
                        std::vector<std::vector<int64_t>> input_shapes, std::vector<std::vector<int64_t>> output_shapes) {
    std::cout << MAGENTA << "-I-----------------------------------------------" << std::endl;
    std::cout << "-I-  ONNX Name                                    " << std::endl;
    std::cout << "-I-----------------------------------------------" << std::endl;
    std::cout << "-I-  " << onnx_file.substr(0, onnx_file.find(".")) << std::endl;
    std::cout << "-I-----------------------------------------------" << std::endl;
    for (auto const& input: input_shapes) {
        std::cout << "-I-  Input shape NCHW:  (" << input[0] << ", " <<
        input[1] << ", " << input[2] << ", " << input[3] << ")" << std::endl;
    }
    std::cout << "-I-----------------------------------------------" << std::endl;
    for (auto const& output: output_shapes) {
        std::cout << "-I-  Output shape NCHW:  (" << output[0] << ", " <<
        output[1] << ", " << output[2] << ", " << output[3] << ")" << std::endl;
    }
    std::cout << "-I-----------------------------------------------" << std::endl;
    std::cout << "\n-I-----------------------------------------------" << std::endl;
    std::cout << "-I-  Hailo Network Name                           " << std::endl;
    std::cout << "-I-----------------------------------------------" << std::endl;
    std::cout << "-I-  " << hef_file.substr(0, hef_file.find(".")) << std::endl;
    std::cout << "-I-----------------------------------------------" << std::endl;
    for (auto const& value: vstreams.first) {
        std::cout << "-I-  Input shape NHWC:  (1, " << value.get_info().shape.height << ", " <<
        value.get_info().shape.width << ", " << value.get_info().shape.features << ")" << std::endl;
    }
    std::cout << "-I-----------------------------------------------" << std::endl;
    for (auto const& value: vstreams.second) {
        std::cout << "-I-  Output shape NHWC: (1, " << value.get_info().shape.height << ", " <<
        value.get_info().shape.width << ", " << value.get_info().shape.features << ")" << std::endl;
    }
    std::cout << "-I-----------------------------------------------\n" << std::endl << RESET;
}


Expected<std::shared_ptr<ConfiguredNetworkGroup>> configure_network_group(Device &device, const std::string &hef_file) {
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


std::string getCmdOption(int argc, char *argv[], const std::string &option) {
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
    if (cmd.empty() && option=="-num=")
        return "100";
    return cmd;
}


int main(int argc, char**argv) {
    std::chrono::duration<double> total_time;
    std::chrono::time_point<std::chrono::system_clock> total_time_start = std::chrono::high_resolution_clock::now();

    std::string hef_file      = getCmdOption(argc, argv, "-hef=");
    std::string onnx_file      = getCmdOption(argc, argv, "-onnx=");
    std::size_t num_of_frames = stoi(getCmdOption(argc, argv, "-num="));
    std::string image_path      = getCmdOption(argc, argv, "-image=");


    Ort::Env env;
    Ort::AllocatorWithDefaultOptions ort_alloc;
    Ort::Session session{env, onnx_file.c_str(), Ort::SessionOptions{}};

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;

    size_t input_num = session.GetInputCount();
    size_t output_num = session.GetOutputCount();

    for (size_t i = 0; i < input_num; i++){
        input_shapes.push_back(session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }
    for (size_t i = 0; i < output_num; i++){
        output_shapes.push_back(session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }

    std::chrono::time_point<std::chrono::system_clock> start_time_vec;
    std::chrono::duration<double> elapsed_time_s_vec;
    std::chrono::duration<double> onnx_elapsed_time_s_vec;

    auto all_devices          = Device::scan_pcie();
    std::cout << BOLDBLUE << "-I- num_of_frames: " << num_of_frames << std::endl << RESET;

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

    auto input_vstream_params = network_group.value()->make_input_vstream_params(false, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    if (!input_vstream_params){
        std::cerr << "-E- Failed make_input_vstream_params " << input_vstream_params.status() << std::endl;
        return input_vstream_params.status();
    }

    auto output_vstream_params = network_group.value()->make_output_vstream_params(false, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    if (!output_vstream_params){
        std::cerr << "-E- Failed make_output_vstream_params " << output_vstream_params.status() << std::endl;
        return output_vstream_params.status();
    }


    auto input_vstreams  = VStreamsBuilder::create_input_vstreams(*network_group.value(), input_vstream_params.value());
    if (!input_vstreams){
        std::cerr << "-E- Failed create_input_vstreams " << output_vstream_params.status() << std::endl;
        return input_vstreams.status();
    }
    auto output_vstreams = VStreamsBuilder::create_output_vstreams(*network_group.value(), output_vstream_params.value());
    if (!input_vstreams or !output_vstreams) {
        std::cerr << "-E- Failed creating input: " << input_vstreams.status() << " output status:" << output_vstreams.status() << std::endl;
        return input_vstreams.status();
    }

    auto vstreams = std::make_pair(input_vstreams.release(), output_vstreams.release());
    print_net_banner(onnx_file, hef_file, vstreams, input_shapes, output_shapes);

    auto activated_network_group = network_group.value()->activate();
    if (!activated_network_group) {
        std::cerr << "-E- Failed activated network group " << activated_network_group.status();
        return activated_network_group.status();
    }

    cv::Mat imageBGR = cv::imread(image_path, cv::ImreadModes::IMREAD_COLOR); 
    cv::Mat resizedImageBGR, resizedImageRGB, image;
    int height = vstreams.first[0].get_info().shape.height;
    int width = vstreams.first[0].get_info().shape.width;
    
    cv::resize(imageBGR, resizedImageBGR, cv::Size(height, width), cv::InterpolationFlags::INTER_CUBIC);
    cv::cvtColor(resizedImageBGR, resizedImageRGB, cv::ColorConversionCodes::COLOR_BGR2RGB);
    resizedImageRGB.convertTo(image, CV_32F, 1.0 / 255.0);

    auto status  = infer(vstreams.first, vstreams.second, num_of_frames, start_time_vec, elapsed_time_s_vec, onnx_elapsed_time_s_vec, ort_alloc, session, image);

    print_inference_statistics(num_of_frames, elapsed_time_s_vec.count(), onnx_elapsed_time_s_vec.count());

    std::chrono::time_point<std::chrono::system_clock> total_time_end = std::chrono::high_resolution_clock::now();
    total_time = total_time_end - total_time_start;

    if (HAILO_SUCCESS != status) {
        std::cerr << "-E- Inference failed "  << status << std::endl;
        return status;
    }

    std::cout << BOLDBLUE << "-I- Total inference run time: " << (double)total_time.count() << " sec" << RESET << std::endl;
    return HAILO_SUCCESS;
}