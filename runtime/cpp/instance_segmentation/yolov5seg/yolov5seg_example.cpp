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
 * @ file semseg
 * This example demonstrates using virtual streams over c++
 **/

#include "hailo/hailort.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>

using hailort::Device;
using hailort::Hef;
using hailort::Expected;
using hailort::make_unexpected;
using hailort::ConfiguredNetworkGroup;
using hailort::VStreamsBuilder;
using hailort::InputVStream;
using hailort::OutputVStream;
using hailort::MemoryView;

void print_fps(std::int64_t duration, std::string video_path) {
    cv::VideoCapture capture(video_path);
    int count = capture.get(cv::CAP_PROP_FRAME_COUNT);
    double fps = (double)count / (double)duration;
    std::cout << "-I---------------------------------------------------------------------" << std::endl;
    std::cout << "-I- Video FPS: " << fps << std::endl;
    std::cout << "-I---------------------------------------------------------------------" << std::endl;
    capture.release();
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
    return cmd;
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

template <typename T> hailo_status write_all(std::vector<InputVStream> &input, std::string &video_path,  
                                            int height, int width, int channels, std::vector<cv::Mat>& frames) {
    std::cout << "-I- Started write thread " << video_path << std::endl;
    cv::VideoCapture capture(video_path);
    int i=0; 
    cv::Mat frame;
    if(!capture.isOpened())
        throw "Unable to read video file";
    for( ; ; ) {
        capture >> frame;
        if(frame.empty()) {
            break;
        }
        
        if (frames[i].channels() == 3){
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        }

        if (frame.rows != height || frame.cols != width){
            cv::resize(frame, frame, cv::Size(width, height), cv::INTER_AREA);
        }

        frames[i] = frame.clone();
        auto status = input[0].write(MemoryView(frame.data, height * width * channels)); // Writing height * width, 3 channels of uint8
        if (HAILO_SUCCESS != status) 
            return status;   

        i++; 
    }
    std::cout << "-I- Finished write thread " << video_path << std::endl;
    return HAILO_SUCCESS;
}

template <typename T> std::vector<hailo_detection_with_byte_mask_t> convert_nms_with_byte_mask_buffer_to_detections(std::vector<T> &src_buffer)
{
    std::vector<hailo_detection_with_byte_mask_t> detections;
    uint8_t *src_ptr = static_cast<uint8_t*>(src_buffer.data());
    uint16_t detections_count = *(uint16_t*)src_ptr;
    detections.reserve(detections_count);
    size_t buffer_offset = sizeof(uint16_t);
    for (size_t i = 0; i < detections_count; i++) {
        hailo_detection_with_byte_mask_t detection = *(hailo_detection_with_byte_mask_t*)(src_ptr + buffer_offset);
        buffer_offset += sizeof(hailo_detection_with_byte_mask_t) + detection.mask_size;
        detections.emplace_back(std::move(detection));
    }
    return detections;
}

template <typename T> cv::Mat semseg_post_process(std::vector<T>& logits, int width, int height, cv::Mat& frame) {
    std::vector<hailo_detection_with_byte_mask_t> detections = convert_nms_with_byte_mask_buffer_to_detections(logits);
    cv::Mat overlay = cv::Mat::zeros(height, width, CV_8UC3);

    for(const auto& detection : detections) {
        int box_width = (detection.box.x_max - detection.box.x_min) * width + 1;
        int box_height = (detection.box.y_max - detection.box.y_min) * height + 1;

        if (box_width <= width && box_height <= height ) {
            for (size_t i = 0; i < static_cast<size_t>(box_height); ++i) {
                for (size_t j = 0; j < static_cast<size_t>(box_width); ++j) {
                    auto cropped_mask_idx = static_cast<size_t>(i) * box_width + j;
                    if (detection.mask[cropped_mask_idx]) {
                        int overlayX = static_cast<int>(j) + static_cast<int>(detection.box.x_min * width);
                        int overlayY = static_cast<int>(i) + static_cast<int>(detection.box.y_min * height);
                        if (overlayX >= 0 && overlayX < width && overlayY >= 0 && overlayY < height) {
                            overlay.at<cv::Vec3b>(overlayY, overlayX) = cv::Vec3b(30, 255, 255);
                        }
                    }
                }
            }
        cv::rectangle(frame, cv::Rect(detection.box.x_min * width, detection.box.y_min * height, box_width, box_height), cv::Scalar(30, 255, 255), 2);
        }
    }
    cv::addWeighted(frame, 1, overlay, 0.3, 0.0, frame);
    overlay.release();
    return frame;
}

template <typename T> hailo_status read_all(OutputVStream &output, std::string &video_path, int video_height, int video_width,  int height, int width, int frame_count, std::vector<cv::Mat>& frames) {
    std::vector<T> data(output.get_frame_size());
    std::cout << "-I- Started read thread " << video_path << std::endl;
    cv::VideoWriter video("./processed_video.mp4",cv::VideoWriter::fourcc('m','p','4','v'),30, cv::Size(video_width, video_height));

    if (!video.isOpened()) {
        std::cerr << "Error: Unable to open video file for writing." << std::endl;
    }

    for (int i = 0 ; i < frame_count; i++) {
        if(frames[i].size().empty() && i != 0){
            break;
        }
        auto status = output.read(MemoryView(data.data(), data.size()));
        if (HAILO_SUCCESS != status)
            return status;

        auto seg_image = semseg_post_process<T>(data, width, height, frames[i]);
        cv::resize(seg_image, seg_image, cv::Size(video_width, video_height));
        video.write(seg_image);
        seg_image.release();
        frames[i].release();
    }
    video.release();
    std::cout << "-I- Finished read thread " << video_path << std::endl;
    return HAILO_SUCCESS;
}

void print_net_banner(std::pair< std::vector<InputVStream>, std::vector<OutputVStream> > &vstreams) {
    std::cout << "-I---------------------------------------------------------------------" << std::endl;
    std::cout << "-I- Dir  Name                                                          " << std::endl;
    std::cout << "-I---------------------------------------------------------------------" << std::endl;
    for (auto &value: vstreams.first){
        std::cout << "-I- IN:  " << value.get_info().name << std::endl;
    }
    std::cout << "-I---------------------------------------------------------------------" << std::endl;
    for (auto &value: vstreams.second){
    std::cout << "-I- OUT: " << value.get_info().name << std::endl;
    }
    std::cout << "-I---------------------------------------------------------------------\n" << std::endl;
}

template <typename IN_T, typename OUT_T> hailo_status infer(std::vector<InputVStream> &inputs, std::vector<OutputVStream> &outputs, 
                                                            std::string video_path) {
    hailo_status input_status = HAILO_UNINITIALIZED;
    hailo_status output_status = HAILO_UNINITIALIZED;
    std::vector<std::thread> output_threads;
    cv::Mat frame;
    
    cv::VideoCapture capture(video_path);
    capture >> frame;
    int video_height = frame.rows;
    int video_width = frame.cols;
    
    if (!capture.isOpened()){
        throw "Error when reading video";
    }
    int frame_count = (int)capture.get(cv::CAP_PROP_FRAME_COUNT);
    std::vector<cv::Mat> frames((int)frame_count);
    capture.release();

    int input_height = inputs.front().get_info().shape.height;
    int input_width = inputs.front().get_info().shape.width;    

    int input_channels = inputs.front().get_info().shape.features;
    std::thread input_thread([&inputs, &video_path, &input_height, &input_width, &input_channels, &input_status, &frames]() { 
                            input_status = write_all<IN_T>(inputs, video_path, input_height, input_width, input_channels, std::ref(frames)); 
                            });
        
    for (auto &output: outputs){
        output_threads.push_back( std::thread([&output, &video_path, &video_height, &video_width, &input_height, &input_width, &output_status, &frame_count, &frames]() { 
                            output_status = read_all<OUT_T>(output, video_path, video_height, video_width, input_height, input_width, frame_count, std::ref(frames)); 
                            }) );
    }
    
    input_thread.join();
    
    for (auto &out: output_threads)
        out.join();

    if ((HAILO_SUCCESS != input_status) || (HAILO_SUCCESS != output_status)) {
        return HAILO_INTERNAL_FAILURE;
    }

    std::cout << "\n-I- Inference finished successfully\n" << std::endl;
    return HAILO_SUCCESS;
}


int main(int argc, char** argv) {
    std::string hef_file   = getCmdOption(argc, argv, "-hef=");
    std::string video_path = getCmdOption(argc, argv, "-path=");
    auto all_devices       = Device::scan_pcie();
    std::cout << "-I- video path: " << video_path << std::endl;
    std::cout << "-I- hef: " << hef_file << "\n" << std::endl;

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
    if (!input_vstream_params){
        std::cerr << "-E- Failed make_input_vstream_params " << input_vstream_params.status() << std::endl;
        return input_vstream_params.status();
    }

    auto output_vstream_params = network_group.value()->make_output_vstream_params(true, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
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

    print_net_banner(vstreams);

    auto activated_network_group = network_group.value()->activate();
    if (!activated_network_group) {
        std::cerr << "-E- Failed activated network group " << activated_network_group.status();
        return activated_network_group.status();
    }
    
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    auto status  = infer<uint8_t, uint8_t>(vstreams.first, vstreams.second, video_path);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::int64_t duration = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
    print_fps(duration, video_path);
    if (HAILO_SUCCESS != status) {
        std::cerr << "-E- Inference failed "  << status << std::endl;
        return status;
    }

    return HAILO_SUCCESS;
}