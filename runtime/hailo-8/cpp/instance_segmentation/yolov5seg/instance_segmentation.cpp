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
#include "yolov5seg_postprocess.hpp"
#include <cstring> // for std::memcpy

namespace fs = std::filesystem;

/////////// Constants ///////////
constexpr size_t MAX_QUEUE_SIZE = 60;

std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue =
    std::make_shared<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>>(MAX_QUEUE_SIZE);

std::shared_ptr<BoundedTSQueue<InferenceResult>> results_queue =
    std::make_shared<BoundedTSQueue<InferenceResult>>(MAX_QUEUE_SIZE);

void preprocess_callback(const std::vector<cv::Mat>& org_frames, 
                                        std::vector<cv::Mat>& preprocessed_frames, 
                                        uint32_t target_width, uint32_t target_height) {
    preprocessed_frames.reserve(preprocessed_frames.size() + org_frames.size());
    for (const auto &frame : org_frames) {
        cv::Mat f = frame;
        preprocessed_frames.push_back(
            pad_frame_letterbox(f, static_cast<int>(target_height), static_cast<int>(target_width)));
    }
}

// Assumes the packed NMS+mask buffer is outputs[0].
PostprocessCallback make_instseg_postprocess_cb(const int model_w, const int model_h)
{
    // Capture model_w/h; draw in model space, then map back (inverse letterbox)
    return [model_w, model_h](cv::Mat &frame_to_draw,
        const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &outputs)
    {
        const uint8_t *src_ptr = outputs[0].first;
        LetterboxMap map{};
        cv::Mat model_space = make_model_space_canvas(frame_to_draw, model_w, model_h, map);
        // 2) Draw detections + masks in model space from the packed buffer
        draw_detections_and_mask(src_ptr, model_w, model_h, model_space);

        // 3) Inverse letterbox back to original frame size
        map_model_to_frame(model_space, map, frame_to_draw);
    };
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

    auto instseg_postprocess_cb = make_instseg_postprocess_cb(model.get_model_shape().width, model.get_model_shape().height);
    
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
                                instseg_postprocess_cb);

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