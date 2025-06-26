/**
 * Copyright (c) 2020-2025 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
/**
 * @file async_infer_basic_example.cpp
 * This example demonstrates the Async Infer API usage with a specific model.
 **/

#include "async_inference.hpp"
#include "utils.hpp"


/////////// Constants ///////////
constexpr size_t MAX_QUEUE_SIZE = 60;
/////////////////////////////////

std::shared_ptr<BoundedTSQueue<PreprocessedFrameItem>> preprocessed_queue =
    std::make_shared<BoundedTSQueue<PreprocessedFrameItem>>(MAX_QUEUE_SIZE);

std::shared_ptr<BoundedTSQueue<InferenceOutputItem>>   results_queue =
    std::make_shared<BoundedTSQueue<InferenceOutputItem>>(MAX_QUEUE_SIZE);

void release_resources(cv::VideoCapture &capture, cv::VideoWriter &video, InputType &input_type){
    if (input_type.is_video) {
        video.release();
    }
    if (input_type.is_camera) {
        capture.release();
        cv::destroyAllWindows();
    }
    preprocessed_queue->stop();
    results_queue->stop();
}

hailo_status run_post_process(
    InputType &input_type,
    CommandLineArgs args,
    int org_height,
    int org_width,
    size_t frame_count,
    cv::VideoCapture &capture,
    size_t class_count = 80,
    double fps = 30)
    {
    cv::VideoWriter video;
    if (input_type.is_video || (input_type.is_camera && args.save)) {    
        init_video_writer("./processed_video.mp4", video, fps, org_width, org_height);
    }
    int i = 0;
    while (true) {
        show_progress(input_type, i, frame_count);
        InferenceOutputItem output_item;
        if (!results_queue->pop(output_item)) {
            break;
        }
        auto& frame_to_draw = output_item.org_frame;
        auto bboxes = parse_nms_data(output_item.output_data_and_infos[0].first, class_count);
        draw_bounding_boxes(frame_to_draw, bboxes);
        
        if (input_type.is_video || (input_type.is_camera && args.save)) {
            video.write(frame_to_draw);
        }
        if (!show_frame(input_type, frame_to_draw)) {
            break; // break the loop if input is from camera and user pressed 'q' 
        }     
        else if (input_type.is_image || input_type.is_directory) {
            cv::imwrite("processed_image_" + std::to_string(i) + ".jpg", frame_to_draw);
            if (input_type.is_image) {break;}
            else if (input_type.directory_entry_count - 1 == i) {break;}
        }
        i++;
    }
    release_resources(capture, video, input_type);
    return HAILO_SUCCESS;
}

void preprocess_video_frames(cv::VideoCapture &capture,
                          uint32_t width, uint32_t height) {
    cv::Mat org_frame;
    while (true) {
        capture >> org_frame;
        if (org_frame.empty()) {
            preprocessed_queue->stop();
            break;
        }        
        auto preprocessed_frame_item = create_preprocessed_frame_item(org_frame, width, height);
        preprocessed_queue->push(preprocessed_frame_item);
    }
}
void preprocess_image_frames(const std::string &input_path,
                          uint32_t width, uint32_t height) {
    cv::Mat org_frame = cv::imread(input_path);
    auto preprocessed_frame_item = create_preprocessed_frame_item(org_frame, width, height);
    preprocessed_queue->push(preprocessed_frame_item);
}
void preprocess_directory_of_images(const std::string &input_path,
                                uint32_t width, uint32_t height) {
    for (const auto &entry : fs::directory_iterator(input_path)) {
            preprocess_image_frames(entry.path().string(), width, height);
    }
}

hailo_status run_preprocess(CommandLineArgs args, AsyncModelInfer &model, 
                            InputType &input_type, cv::VideoCapture &capture) {

    auto model_input_shape = model.get_infer_model()->hef().get_input_vstream_infos().release()[0].shape;
    uint32_t target_height = model_input_shape.height;
    uint32_t target_width = model_input_shape.width;
    print_net_banner(get_hef_name(args.detection_hef), std::ref(model.get_inputs()), std::ref(model.get_outputs()));

    if (input_type.is_image) {
        preprocess_image_frames(args.input_path, target_width, target_height);
    }
    else if (input_type.is_directory) {
        preprocess_directory_of_images(args.input_path, target_width, target_height);
    }
    else{
        preprocess_video_frames(capture, target_width, target_height);
    } 
    return HAILO_SUCCESS;
}

hailo_status run_inference_async(AsyncModelInfer& model,
                            std::chrono::duration<double>& inference_time) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        PreprocessedFrameItem item;
        if (!preprocessed_queue->pop(item)) {
            break;
        }

        // Pass as parameters the device input and a lambda that captures the original frame and uses the provided output buffers.
        model.infer(
                    std::make_shared<cv::Mat>(item.resized_for_infer),
                    [org_frame = item.org_frame, queue = results_queue](
                        const hailort::AsyncInferCompletionInfo &info,
                        const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &output_data_and_infos,
                        const std::vector<std::shared_ptr<uint8_t>> &output_guards)
                    {
                        InferenceOutputItem output_item;
                        output_item.org_frame = org_frame;
                        output_item.output_data_and_infos = output_data_and_infos;
                        output_item.output_guards = output_guards;  // <-- Add this field to the struct!
                        queue->push(output_item);
                    });
    }
    results_queue->stop();
    auto end_time = std::chrono::high_resolution_clock::now();

    inference_time = end_time - start_time;

    return HAILO_SUCCESS;
}

int main(int argc, char** argv)
{
    size_t class_count = 80; // 80 classes in COCO dataset
    double fps = 30;

    std::chrono::duration<double> inference_time;
    std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::high_resolution_clock::now();
    double org_height, org_width;
    cv::VideoCapture capture;
    size_t frame_count;
    InputType input_type;

    CommandLineArgs args = parse_command_line_arguments(argc, argv);
    AsyncModelInfer model(args.detection_hef);
    input_type = determine_input_type(args.input_path, std::ref(capture), org_height, org_width, frame_count);

    auto preprocess_thread = std::async(run_preprocess,
                                        args,
                                        std::ref(model),
                                        std::ref(input_type),
                                        std::ref(capture));

    auto inference_thread = std::async(run_inference_async,
                                    std::ref(model),
                                    std::ref(inference_time));

    auto output_parser_thread = std::async(run_post_process,
                                std::ref(input_type),
                                args,
                                org_height,
                                org_width,
                                frame_count,
                                std::ref(capture),
                                class_count,
                                fps);

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