/**
 * Copyright (c) 2020-2023 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
/**
 * @file async_infer_basic_example.cpp
 * This example demonstrates the Async Infer API usage with a specific model.
 **/

#include "hailo/hailort.hpp"
#include "common.h"
#include "common/yolo_hailortpp.hpp"
#include "common/labels/coco_ninety.hpp"

#include <iostream>
#include <future>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>

#if defined(__unix__)
#include <sys/mman.h>
#endif

std::mutex m;

void print_inference_statistics(std::chrono::duration<double> inference_time,
                                std::string hef_file, double frame_count) { 
    std::cout << BOLDGREEN << "\n-I-----------------------------------------------" << std::endl;
    std::cout << "-I- " << hef_file.substr(0, hef_file.find(".")) << std::endl;
    std::cout << "-I-----------------------------------------------" << std::endl;
    std::cout << "\n-I-----------------------------------------------" << std::endl;
    std::cout << "-I- Inference & Postprocess                        " << std::endl;
    std::cout << "-I-----------------------------------------------" << std::endl;
    std::cout << "-I- Average FPS:  " << frame_count / (inference_time.count()) << std::endl;
    std::cout << "-I- Total time:   " << inference_time.count() << " sec" << std::endl;
    std::cout << "-I- Latency:      " << 1.0 / (frame_count / (inference_time.count()) / 1000) << " ms" << std::endl;
    std::cout << "-I-----------------------------------------------" << std::endl;
}

using namespace hailort;

static std::shared_ptr<uint8_t> page_aligned_alloc(size_t size, void* buff = nullptr)
{
#if defined(__unix__)
    auto addr = mmap(buff, size, PROT_WRITE | PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (MAP_FAILED == addr) throw std::bad_alloc();
    return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(addr), [size](void *addr) { munmap(addr, size); });
#elif defined(_MSC_VER)
    auto addr = VirtualAlloc(buff, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!addr) throw std::bad_alloc();
    return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(addr), [](void *addr){ VirtualFree(addr, 0, MEM_RELEASE); });
#else
#pragma error("Aligned alloc not supported")
#endif
}


hailo_status run_postprocess_and_visualization(std::vector<cv::Mat>& frames, size_t frame_count,
                                    TSQueue<std::vector<std::pair<uint8_t*, hailo_vstream_info_t>>>& inferred_data_queue,
                                    double org_height, double org_width){

    // cv::VideoWriter video("./processed_video.mp4", cv::VideoWriter::fourcc('m','p','4','v'),30, cv::Size((int)org_width, (int)org_height));

    for (size_t i = 0; i < frame_count; i++){

        auto output_data_and_infos = inferred_data_queue.pop();

        HailoROIPtr roi = std::make_shared<HailoROI>(HailoROI(HailoBBox(0.0f, 0.0f, 1.0f, 1.0f)));
        
        for (auto& data_and_info : output_data_and_infos) {
            roi->add_tensor(std::make_shared<HailoTensor>(reinterpret_cast<uint8_t*>(data_and_info.first), data_and_info.second));
        }

        filter(roi);

        std::vector<HailoDetectionPtr> detections = hailo_common::get_hailo_detections(roi);

        for (auto &detection : detections) {
            if (detection->get_confidence()==0) {
                continue;
            }

            HailoBBox bbox = detection->get_bbox();

            cv::rectangle(frames[0], cv::Point2f(bbox.xmin() * float(org_width), bbox.ymin() * float(org_height)), 
                        cv::Point2f(bbox.xmax() * float(org_width), bbox.ymax() * float(org_height)), 
                        cv::Scalar(0, 0, 255), 1);
            
            std::cout << "Detection: " << detection->get_label() << ", Confidence: " << std::fixed << 
                                std::setprecision(2) << detection->get_confidence() * 100.0 << "%" << std::endl;

        }
        // cv::imshow("Display window", frames[0]);
        // cv::waitKey(0);

        // video.write(frames[0]);
        // cv::imwrite("output_image.jpg", frames[0]);

        m.try_lock();
        frames[0].release();
        frames.erase(frames.begin());
        m.unlock();
    }

    // video.release();

    return HAILO_SUCCESS;
}


hailo_status run_inference(std::shared_ptr<hailort::InferModel> infer_model,
                            std::vector<std::promise<cv::Mat>>& frames_promises,
                            std::vector<std::future<cv::Mat>>& frames_futures,
                            size_t frame_count,
                            std::vector<std::shared_ptr<cv::Mat>>& input_buffer_guards,
                            std::vector<std::shared_ptr<uint8_t>>& output_buffer_guards,
                            std::vector<std::string>& output_names,
                            TSQueue<std::vector<std::pair<uint8_t*, hailo_vstream_info_t>>>& inferred_data_queue,
                            std::chrono::duration<double>& inference_time) {

    auto configured_infer_model = infer_model->configure();
    if (!configured_infer_model) {
        std::cerr << "Failed to create configured infer model, status = " << configured_infer_model.status() << std::endl;
        return configured_infer_model.status();
    }

    AsyncInferJob last_infer_job;

    std::shared_ptr<uint8_t> output_buffer;

    auto bindings = configured_infer_model->create_bindings();
    if (!bindings) {
        std::cerr << "Failed to create infer bindings, status = " << bindings.status() << std::endl;
        return bindings.status();
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < frame_count; i++){

        auto preprocessed_image = std::make_shared<cv::Mat>(frames_futures[i].get());

        for (const auto &input_name : infer_model->get_input_names()) {
            size_t input_frame_size = infer_model->input(input_name)->get_frame_size();
            auto status = bindings->input(input_name)->set_buffer(MemoryView(preprocessed_image->data, input_frame_size));
            if (HAILO_SUCCESS != status) {
                std::cerr << "Failed to set infer input buffer, status = " << status << std::endl;
                return status;
            }

            input_buffer_guards.push_back(preprocessed_image);
        }

        std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> output_data_and_infos;
        
        for (const auto &output_name : output_names) {
            size_t output_frame_size = infer_model->output(output_name)->get_frame_size();
            output_buffer = page_aligned_alloc(output_frame_size);
            auto status = bindings->output(output_name)->set_buffer(MemoryView(output_buffer.get(), output_frame_size));
            if (HAILO_SUCCESS != status) {
                std::cerr << "Failed to set infer output buffer, status = " << status << std::endl;
                return status;
            }

            output_data_and_infos.push_back(std::make_pair(
                                                            bindings->output(output_name)->get_buffer()->data(), 
                                                            infer_model->hef().get_output_vstream_infos().release()[0]
                                                        ));

            output_buffer_guards.push_back(output_buffer);
        }

        auto status = configured_infer_model->wait_for_async_ready(std::chrono::milliseconds(1000));
        if (HAILO_SUCCESS != status) {
            std::cerr << "Failed to run wait_for_async_ready, status = " << status << std::endl;
            return status;
        }

        auto job = configured_infer_model->run_async(bindings.value(), 
                                                        [&inferred_data_queue, output_data_and_infos, output_buffer](const hailort::AsyncInferCompletionInfo& info){
            inferred_data_queue.push(output_data_and_infos);
            (void)output_buffer;
        });

        if (!job) {
            std::cerr << "Failed to start async infer job, status = " << job.status() << std::endl;
            return job.status();
        }

        job->detach();
        if (i == frame_count - 1) {
            last_infer_job = job.release();
        }
    }

    auto status = last_infer_job.wait(std::chrono::milliseconds(1000));
    if (HAILO_SUCCESS != status) {
        std::cerr << "Failed to wait for infer to finish, status = " << status << std::endl;
        return status;
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    inference_time = end_time - start_time;

    return HAILO_SUCCESS;
}


void use_single_frame(std::vector<cv::Mat>& frames, 
                    std::vector<std::promise<cv::Mat>>& frames_promises,
                    cv::Mat& image, int frame_count, cv::Size shape){

    for(int i = 0; i < frame_count; i++) {
        m.lock();
        frames.push_back(image);
        m.unlock();

        cv::resize(image, image, shape, 1);

        frames_promises[i].set_value(image);
    }
}

hailo_status run_preprocess(std::vector<cv::Mat>& frames, std::vector<std::promise<cv::Mat>>& frames_promises,
                            std::string input_path, uint32_t height, uint32_t width, std::string& cmd_num_frames) {

    cv::VideoCapture capture;
    if (input_path.empty()) {
        capture.open(0, cv::CAP_ANY);
        if (!capture.isOpened()) {
            throw "Unable to read camera input";
        }
    }
    else{
        capture.open(input_path, cv::CAP_ANY);
        if(!capture.isOpened())
            throw "Unable to read input file";
    }

    cv::Mat org_frame; 

    if (input_path.find(".avi") == std::string::npos && input_path.find(".mp4") == std::string::npos){
        capture >> org_frame;
        int frame_count = cmd_num_frames.empty() ? 1 : std::stoi(cmd_num_frames);
        use_single_frame(frames, frames_promises, std::ref(org_frame), frame_count, cv::Size(width, height));
        org_frame.release();
        capture.release();
    }
    else {
        int i = 0;
        for(;;) {
            capture >> org_frame;
            if(org_frame.empty()) {
                break;
            }

            m.lock();
            frames.push_back(org_frame);
            m.unlock();
                
            cv::resize(org_frame, org_frame, cv::Size(width, height), 1);

            frames_promises[i].set_value(org_frame);

            i++;

        }
    }

    capture.release();

    return HAILO_SUCCESS;
}


hailo_status configure_and_infer(std::shared_ptr<hailort::InferModel> infer_model, std::string input_path, 
                                std::chrono::time_point<std::chrono::system_clock>& start_time,
                                std::chrono::time_point<std::chrono::system_clock>& end_time, 
                                std::chrono::duration<double>& inference_time, size_t frame_count,
                                double org_height, double org_width, std::string cmd_img_num){

    std::vector<cv::Mat> frames;

    // The buffers are stored here as a guard for the memory. The buffer will be freed only after
    // configured_infer_model will be released.
    std::vector<std::shared_ptr<cv::Mat>> input_buffer_guards;
    std::vector<std::shared_ptr<uint8_t>> output_buffer_guards;

    input_buffer_guards.reserve(infer_model->inputs().size());
    output_buffer_guards.reserve(infer_model->outputs().size());

    std::vector<std::string> output_names = infer_model->get_output_names();

    std::vector<std::promise<cv::Mat>> frames_promises(frame_count);
    std::vector<std::future<cv::Mat>> frames_futures(frame_count);

    for (size_t i = 0; i < frames_promises.size(); i++) {
        frames_futures[i] = frames_promises[i].get_future();
    }

    TSQueue<std::vector<std::pair<uint8_t*, hailo_vstream_info_t>>> inferred_data_queue;

    auto model_input_shape = infer_model->hef().get_input_vstream_infos().release()[0].shape;

    auto preprocess_thread(std::async(run_preprocess,
                                    std::ref(frames),
                                    std::ref(frames_promises),
                                    input_path, 
                                    model_input_shape.height, 
                                    model_input_shape.width,
                                    std::ref(cmd_img_num)));

    auto inference_thread(std::async(run_inference,
                                    infer_model,
                                    std::ref(frames_promises),
                                    std::ref(frames_futures),
                                    frame_count,
                                    std::ref(input_buffer_guards),
                                    std::ref(output_buffer_guards),
                                    std::ref(output_names),
                                    std::ref(inferred_data_queue),
                                    std::ref(inference_time)));

    auto postprocess_thread(std::async(run_postprocess_and_visualization,
                                    std::ref(frames),
                                    frame_count,
                                    std::ref(inferred_data_queue),
                                    org_height, 
                                    org_width));

    auto preprocess_status = preprocess_thread.get();
    auto inference_status = inference_thread.get();
    auto postprocess_status = postprocess_thread.get();

    if (HAILO_SUCCESS != preprocess_status) {
        std::cerr << "Preprocess failed with status " << preprocess_status << std::endl;
        return preprocess_status; 
    }

    if (HAILO_SUCCESS != inference_status) {
        std::cerr << "Preprocess failed with status " << inference_status << std::endl;
        return inference_status; 
    }

    if (HAILO_SUCCESS != postprocess_status) {
        std::cerr << "Post-processing failed with status " << postprocess_status << std::endl;
        return postprocess_status;
    }

    return HAILO_SUCCESS;
}


void print_net_banner(std::string detection_model_name, 
                    std::vector<hailort::InferModel::InferStream>& inputs, 
                    std::vector<hailort::InferModel::InferStream>& outputs) {
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
    std::cout << BOLDMAGENTA << "-I-  Network  Name                               " << std::endl << RESET;
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
    std::cout << BOLDMAGENTA << "-I   " << detection_model_name                     << std::endl << RESET;
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
    for (auto& input: inputs) {
        auto shape = input.shape();
        std::cout << MAGENTA << "-I-  Input: " << input.name() 
        << ", Shape: (" << shape.height << ", " << shape.width << ", " << shape.features << ")" 
        << std::endl << RESET;
    }
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
    for (auto& output: outputs) {
        auto shape = output.shape();
        std::cout << MAGENTA << "-I-  Output: " << output.name() 
        << ", Shape: (" << shape.height << ", " << shape.width << ", " << shape.features << ")" 
        << std::endl << RESET;
    }
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------\n" << std::endl << RESET;
}

std::string get_hef_name(std::string path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
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
            cmd = arg.substr(found);
            return cmd;
        }
    }
    return cmd;
}


int main(int argc, char** argv)
{

    std::chrono::duration<double> total_time;
    std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::high_resolution_clock::now();

    std::string detection_hef      = getCmdOption(argc, argv, "-hef=");
    std::string input_path      = getCmdOption(argc, argv, "-input=");
    std::string image_num      = getCmdOption(argc, argv, "-num=");

    std::chrono::time_point<std::chrono::system_clock> write_time_vec;
    std::chrono::time_point<std::chrono::system_clock> postprocess_end_time;
    std::chrono::duration<double> inference_time;

    auto vdevice = VDevice::create();
    if (!vdevice) {
        std::cerr << "Failed create vdevice, status = " << vdevice.status() << std::endl;
        return vdevice.status();
    }

    // Create infer model from HEF file.
    auto infer_model_exp = vdevice.value()->create_infer_model(detection_hef);
    if (!infer_model_exp) {
        std::cerr << "Failed to create infer model, status = " << infer_model_exp.status() << std::endl;
        return infer_model_exp.status();
    }
    auto infer_model = infer_model_exp.release();

    std::vector<hailort::InferModel::InferStream> inputs = infer_model->inputs();
    std::vector<hailort::InferModel::InferStream> outputs = infer_model->outputs();

    print_net_banner(get_hef_name(detection_hef), std::ref(inputs), std::ref(outputs));

    cv::VideoCapture capture;
    double frame_count;
    if (input_path.empty()) {
        capture.open(0, cv::CAP_ANY);
        if (!capture.isOpened()) {
            throw "Error in camera input";
        }
        frame_count = -1;
    }
    else {
        capture.open(input_path, cv::CAP_ANY);
        if (!capture.isOpened()){
            throw "Error when reading video";
        }
        if (!image_num.empty()){
            if (input_path.find(".avi") == std::string::npos && input_path.find(".mp4") == std::string::npos){
                frame_count = std::stoi(image_num);
            }
            else {
                frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);
                image_num = "";
            }
        }
        else {
            frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);
        }
    }

    double org_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    double org_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);

    capture.release();

    auto status = configure_and_infer(infer_model, 
                                        input_path, 
                                        write_time_vec, 
                                        postprocess_end_time, 
                                        inference_time, 
                                        (size_t)frame_count, 
                                        org_height, 
                                        org_width,
                                        image_num);

    if (HAILO_SUCCESS != status) {
        std::cerr << "Failed running inference with status = " << status << std::endl;
        return status;
    }

    std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::high_resolution_clock::now();
    total_time = t_end - t_start;
    
    print_inference_statistics(inference_time, detection_hef, (double)frame_count);

    std::cout << BOLDBLUE << "\n-I- Application run finished successfully" << RESET << std::endl;
    std::cout << BOLDBLUE << "-I- Total application run time: " << (double)total_time.count() << " sec" << RESET << std::endl;
    return HAILO_SUCCESS;
}
