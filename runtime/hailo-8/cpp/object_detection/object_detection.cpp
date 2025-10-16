/**
 * Copyright (c) 2020-2025 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
/**
 * @file async_infer_basic_example.cpp
 * This example demonstrates the Async Infer API usage with a specific model.
 **/
#include "toolbox.hpp"
using namespace hailo_utils;

#include "hailo_infer.hpp"
#include "utils.hpp"

namespace fs = std::filesystem;

/////////// Constants ///////////
constexpr size_t MAX_QUEUE_SIZE = 60;

std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue =
    std::make_shared<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>>(MAX_QUEUE_SIZE);

std::shared_ptr<BoundedTSQueue<InferenceResult>> results_queue =
    std::make_shared<BoundedTSQueue<InferenceResult>>(MAX_QUEUE_SIZE);

// Task-specific preprocessing callback
void preprocess_callback(const std::vector<cv::Mat>& org_frames, 
                                        std::vector<cv::Mat>& preprocessed_frames, 
                                        uint32_t target_width, uint32_t target_height) {
    for (const auto& frame : org_frames) {
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(target_width, target_height));
        preprocessed_frames.push_back(resized_frame);
    }
}

// Task-specific postprocessing callback
void postprocess_callback(cv::Mat& frame_to_draw, 
                                         const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>>& output_data_and_infos) {
    size_t class_count = 80; // 80 classes in COCO dataset
    auto bboxes = parse_nms_data(output_data_and_infos[0].first, class_count);
    draw_bounding_boxes(frame_to_draw, bboxes);
}

int main(int argc, char** argv)
{
    double fps = 30;

    std::chrono::duration<double> inference_time;
    std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::high_resolution_clock::now();
    double org_height, org_width;
    cv::VideoCapture capture;
    size_t frame_count;
    InputType input_type;

    CommandLineArgs args = parse_command_line_arguments(argc, argv);
    auto batch_size = std::stoi(args.batch_size);
    HailoInfer model(args.detection_hef, batch_size);
    input_type = determine_input_type(args.input_path, std::ref(capture), org_height, org_width, frame_count, batch_size);

    auto preprocess_thread = std::async(run_preprocess,
                                        args.input_path,
                                        args.detection_hef,
                                        std::ref(model),
                                        std::ref(input_type),
                                        std::ref(capture),
                                        batch_size,
                                        preprocessed_batch_queue,
                                        preprocess_callback);

    auto inference_thread = std::async(run_inference_async,
                                    std::ref(model),
                                    std::ref(inference_time),
                                    preprocessed_batch_queue,
                                    results_queue);

    auto output_parser_thread = std::async(run_post_process,
                                std::ref(input_type),
                                org_height,
                                org_width,
                                frame_count,
                                std::ref(capture),
                                fps,
                                batch_size,
                                results_queue,
                                postprocess_callback);

    hailo_status status = wait_and_check_threads(
        preprocess_thread,    "Preprocess",
        inference_thread,     "Inference",
        output_parser_thread, "Postprocess "
    );
    if (HAILO_SUCCESS != status) {
        return status;
    }

    if(!input_type.is_camera) {
        std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::high_resolution_clock::now();
        print_inference_statistics(inference_time, args.detection_hef, frame_count, t_end - t_start);
    }

    return HAILO_SUCCESS;
}