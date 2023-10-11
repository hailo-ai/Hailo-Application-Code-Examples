#include "tensors_buffers.hpp"
#include "yolo_post.hpp"
#include "hailo/hailort.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <queue>
#include <condition_variable>
#include <memory>
#include <vector>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <mutex>

using namespace hailort;

// #define _EMBEDDED_ // add to enable embedded mode (jpg instead of mp4, for h15 / imx)

#ifndef _EMBEDDED_
#define SAVE_TO_FILE // comment out to disable saving to file
#endif

constexpr auto TIMEOUT = std::chrono::milliseconds(1000);
constexpr int default_max_num_frames_to_process = 1200;

class AbstractCapture {
public:
    int counter_frames;

    AbstractCapture(const std::string& filename, int max_num_frames_to_process, int counter_frames) : filename(filename), max_num_frames_to_process(max_num_frames_to_process), counter_frames(counter_frames) {}
    virtual int getNextFrame(AlignedBuffer buffer) = 0; // Pure virtual function
    virtual hailo_status setHeightWidth(double height_hef, double width_hef) = 0; // Pure virtual function
    int getWidth() {
        return width;
    }
    int getHeight() {
        return height;
    }

protected:
    std::string filename;
    int max_num_frames_to_process;
    int width;
    int height;
};

class ImageCapture : public AbstractCapture {
public:
    ImageCapture(const std::string& filename) : AbstractCapture(filename, default_max_num_frames_to_process, 0) {
        capture = cv::imread(filename);
        if (capture.empty()) {
            std::cerr << "Error: Could not open image file " << filename << std::endl;
            exit(1);
        }
    }

    int getNextFrame(AlignedBuffer buffer) override {
        if (counter_frames < max_num_frames_to_process) {
            cv::Mat frame(height, width, CV_8UC3, static_cast<void*>(buffer.get()));
            capture.copyTo(frame); // TODO: make sure same type and size, so there won't be reallocation
            counter_frames++;
            return EXIT_SUCCESS;
        }
        return EXIT_FAILURE; // finished all frames (images)
    }

    hailo_status setHeightWidth(double height_hef, double width_hef) override {
        cv::Size size = capture.size();
        int height_capture = size.height;
        int width_capture = size.width;
        if (height_hef != height_capture || width_hef != width_capture) {
            std::cerr << "Error: Frame size of HEF and capture device do not match, hxw HEF: " << height_hef << "x" << width_hef << ", hxw capture: " << height_capture << "x" << width_capture << std::endl;
            return HAILO_INVALID_ARGUMENT; // TODO: maybe resize video and not fail, and only print warnning
        }
        height = height_hef;
        width = width_hef;
        return HAILO_SUCCESS;
    }

private:
    cv::Mat capture;

};

class VideoCapture : public AbstractCapture {
public:
    VideoCapture(const std::string& filename) : AbstractCapture(filename, default_max_num_frames_to_process, 0), capture(filename) {
        if (!capture.isOpened()) {
            std::cerr << "Error: Could not open video file " << filename << std::endl;
            exit(1);
        }
        max_num_frames_to_process = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
    }

    int getNextFrame(AlignedBuffer buffer) override {
        if (counter_frames < max_num_frames_to_process) {
            cv::Mat frame(height, width, CV_8UC3, static_cast<void*>(buffer.get()));
            capture >> frame;
            if (!frame.empty()) {
                counter_frames++;
                return EXIT_SUCCESS;
            }
        }
        return EXIT_FAILURE; // finished all frames (video)
    }

    hailo_status setHeightWidth(double height_hef, double width_hef) override {
        double height_capture = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
        double width_capture = capture.get(cv::CAP_PROP_FRAME_WIDTH);
        if (height_hef != height_capture || width_hef != width_capture) {
            std::cerr << "Error: Frame size of HEF and capture device do not match, hxw HEF: " << height_hef << "x" << width_hef << ", hxw capture: " << height_capture << "x" << width_capture << std::endl;
            return HAILO_INVALID_ARGUMENT; // TODO: maybe resize video and not fail, and only print warnning
        }
        height = height_hef;
        width = width_hef;
        return HAILO_SUCCESS;
    }

private:
    cv::VideoCapture capture;

};

class App {
private:
    std::unique_ptr<hailort::Device> device;
    std::shared_ptr<ConfiguredNetworkGroup> network_group; // TODO: all the .get() are because of reference_wrapper, maybe save as plain reference of inputs & outputs vstreams instead (or in addition)
    int num_outputs;
    std::mutex print_mutex;
    std::atomic<int> input_ctr;
    std::array<std::atomic<int>, default_num_outputs> output_ctr;
    std::array<std::atomic<int>, default_num_outputs> output_callback_ctr;
    std::atomic<int> pp_ctr;
    std::atomic<bool> continue_run;
    bool print;
    std::mutex outputs_mutex;
    std::condition_variable outputs_cv;

public:
    std::unique_ptr<AbstractCapture> camera;
    App(const std::string filename) : num_outputs(0), input_ctr(0), pp_ctr(0), continue_run(false), print(false) {
        size_t dot_position = filename.rfind('.');
        if (dot_position != std::string::npos) {
            const std::string extension = filename.substr(dot_position + 1);
            if (extension == "jpg" || extension == "jpeg") {
                camera = std::make_unique<ImageCapture>(filename);
            } else if (extension == "mp4") {
                camera = std::make_unique<VideoCapture>(filename);
            }
            else {
                std::cout << "Unknown file format " << extension << std::endl;
                exit(1);
            }
        }
        else {
            std::cout << "Unknown file format" << std::endl;
            exit(1);
        }
    }

    const int get_num_frames_processed() {
        return camera->counter_frames;
    }

    hailo_status init(const std::string& hef_path, bool print) {
        auto device_exp = Device::create();
        if (!device_exp) {
            std::cerr << "Failed to create device " << device_exp.status() << std::endl;
            return device_exp.status();
        }
        device = device_exp.release();

        auto network_group_exp = configureNetwork(hef_path);
        if (!network_group_exp) {
            std::cerr << "Failed to configure network group" << std::endl;
            return network_group_exp.status();
        }
        network_group = std::move(network_group_exp.value());
        auto status = camera->setHeightWidth(network_group->get_input_streams()[0].get().get_info().hw_shape.height, network_group->get_input_streams()[0].get().get_info().hw_shape.width);
        if (status != HAILO_SUCCESS) {
            std::cerr << "Failed to set input width and height" << std::endl;
            return status;
        }
        num_outputs = network_group->get_output_vstream_infos()->size();
        // TODO: add num_inputs to support multiple inputs
        for (int i = 0; i < num_outputs; i++) {
            output_ctr[i].store(0);
        }
        for (int i = 0; i < num_outputs; i++) {
            output_callback_ctr[i].store(0);
        }

        continue_run = true;
        this->print = print;
        return HAILO_SUCCESS;
    }

    hailort::Expected<std::shared_ptr<hailort::ConfiguredNetworkGroup>> configureNetwork(const std::string& hef_path) {
        auto hef_exp = Hef::create(hef_path);
        if (!hef_exp) {
            return make_unexpected(hef_exp.status());
        }
        auto hef = hef_exp.release();

        auto configure_params = device->create_configure_params(hef);
        if (!configure_params) {
            return make_unexpected(configure_params.status());
        }

        // change stream_params to operate in async mode
        for (auto &ng_name_params_pair : *configure_params) {
            for (auto &stream_params_name_pair : ng_name_params_pair.second.stream_params_by_name) {
                stream_params_name_pair.second.flags = HAILO_STREAM_FLAGS_ASYNC;
            }
        }

        auto network_groups = device->configure(hef, configure_params.value());
        if (!network_groups) {
            return make_unexpected(network_groups.status());
        }

        if (1 != network_groups->size()) {
            std::cerr << "Invalid amount of network groups" << std::endl;
            return make_unexpected(HAILO_INTERNAL_FAILURE);
        }

        return std::move(network_groups->at(0));
    }

    static void input_async_callback(const InputStream::CompletionInfo &completion_info)
    {
        if ((HAILO_SUCCESS != completion_info.status) && (HAILO_STREAM_ABORTED_BY_USER  != completion_info.status)) {
            // We will get HAILO_STREAM_ABORTED_BY_USER  when activated_network_group is destructed.
            std::cerr << "Got an unexpected status on callback. status=" << completion_info.status << std::endl;
        }
    }

    hailo_status run() {
        auto inputs = network_group->get_input_streams();
        auto outputs = network_group->get_output_streams();
        // --------------------------------------------------- create wrappers for input / output -----------------------------------
        InputTensor input_tensor(inputs[0].get().get_info()); // we have only 1 input
        std::vector<hailo_stream_info_t> outputs_stream_infos; // we have n outputs
        for (auto& output : outputs) {
            outputs_stream_infos.push_back(output.get().get_info());
        }
        OutputTensors output_tensors(outputs_stream_infos);
        // --------------------------------------------------- activate network group -----------------------------------------------
        auto activated_network_group = network_group->activate();
        if (!activated_network_group) {
            std::cerr << "Failed to activate network group "  << activated_network_group.status() << std::endl;
            return activated_network_group.status();
        }
        std::chrono::steady_clock::time_point begin_time = std::chrono::steady_clock::now();
        // --------------------------------------------------- input thread -------------------------------------------------------
        std::atomic<hailo_status> input_status(HAILO_UNINITIALIZED);
        std::thread input_thread([&print_mutex=print_mutex, &input_status, &inputs, &camera=camera, &input_tensor, &input_ctr=input_ctr, &continue_run=continue_run, &print=print]() {
            while (continue_run) {
                input_status = inputs[0].get().wait_for_async_ready(inputs[0].get().get_frame_size(), TIMEOUT);
                if (HAILO_SUCCESS != input_status) { return; }
                auto input_buffer = page_aligned_alloc(inputs[0].get().get_frame_size());
                auto get_frame_status = camera->getNextFrame(input_buffer);
                if (EXIT_SUCCESS != get_frame_status) {
                    continue_run = false;
                    return; 
                }
                input_tensor.m_queue.push(input_buffer);
                input_status = inputs[0].get().write_async(input_buffer.get(), inputs[0].get().get_frame_size(), input_async_callback);
                if (HAILO_SUCCESS != input_status) { return; }
                if (print) {
                    std::unique_lock<std::mutex> lock(print_mutex);
                    std::cout << "input async write " << input_ctr << std::endl;
                }
                input_ctr++;
            }
        });
        // --------------------------------------------------- outputs threads -------------------------------------------------------
        std::vector<std::thread> output_threads;
        output_threads.reserve(num_outputs);
        std::vector<std::atomic<hailo_status>> output_statuses(num_outputs);
        for (auto& status : output_statuses) {
            status.store(HAILO_UNINITIALIZED);
        }
        std::vector<OutputStream::TransferDoneCallback> output_async_callbacks_guard;
        for (int i = 0; i < num_outputs; i++) {
            OutputStream::TransferDoneCallback output_async_callback = [&output_callback_ctr=output_callback_ctr, i, &outputs_mutex=outputs_mutex, &outputs_cv=outputs_cv](const OutputStream::CompletionInfo &completion_info) {
                if ((HAILO_SUCCESS != completion_info.status) && (HAILO_STREAM_ABORTED_BY_USER != completion_info.status)) {
                    // We will get HAILO_STREAM_ABORTED_BY_USER when activated_network_group is destructed.
                    std::cerr << "Got an unexpected status on callback. status=" << completion_info.status << std::endl;
                }
                std::unique_lock<std::mutex> lock(outputs_mutex);
                output_callback_ctr[i]++;
                if (output_callback_ctr[0] > 0 && output_callback_ctr[1] > 0 && output_callback_ctr[2] > 0) {
                    outputs_cv.notify_all();
                }
                lock.unlock();

            };
            output_async_callbacks_guard.push_back(output_async_callback);
        }
        for (int i = 0; i < num_outputs; i++) {
            output_threads.emplace_back(std::thread([&print_mutex=print_mutex, &output_statuses, &outputs, i, &output_tensors, &input_ctr=input_ctr, &output_ctr=output_ctr[i], &continue_run=continue_run, &print=print, &output_async_callback=output_async_callbacks_guard[i]]() {
            while (output_ctr < input_ctr || continue_run) { // we haven't finished to process all the input frames
                output_statuses[i] = outputs[i].get().wait_for_async_ready(outputs[i].get().get_frame_size(), TIMEOUT);
                if (HAILO_SUCCESS != output_statuses[i]) { return; }

                auto output_buffer = page_aligned_alloc(outputs[i].get().get_frame_size());
                output_statuses[i] = outputs[i].get().read_async(output_buffer.get(), outputs[i].get().get_frame_size(), output_async_callback);
                if (HAILO_SUCCESS != output_statuses[i]) { return; }
                output_tensors.outputs[i].get()->m_queue.push(output_buffer);
                if (print) {
                    std::unique_lock<std::mutex> lock(print_mutex);
                    std::cout << "output async read " << output_ctr << ", thread " << i << std::endl;
                }
                output_ctr++;
                }
            }));
        }
        // --------------------------------------------------- post-process thread -------------------------------------------------------
        std::atomic<hailo_status> pp_status(HAILO_UNINITIALIZED);
        std::thread pp_thread([&print_mutex=print_mutex, &camera=camera, &input_tensor, &output_tensors, &input_ctr=input_ctr, &output_callback_ctr=output_callback_ctr, &pp_ctr=pp_ctr, &continue_run=continue_run, &print=print, &pp_status=pp_status, &outputs_mutex=outputs_mutex, &outputs_cv=outputs_cv]() {
#ifdef SAVE_TO_FILE
            cv::VideoWriter video("./processed_video.mp4", cv::VideoWriter::fourcc('m','p','4','v'),30, cv::Size(camera->getWidth(), camera->getHeight())); // add in order to save to file the processed video
#endif
            while (pp_ctr < input_ctr || continue_run) { // we haven't finished to process all the input frames // was: while (pp_ctr < input_ctr || continue_run)
                // we have to make sure that all 3 outputs finished calback before we can pop them 
                std::unique_lock<std::mutex> lock(outputs_mutex);
                while (output_callback_ctr[0] <= 0 || output_callback_ctr[1] <= 0 || output_callback_ctr[2] <= 0) {
                    outputs_cv.wait(lock); // Release the lock and wait for a signal.
                }
                // was: lock.unlock() here

                auto out_0 = output_tensors.outputs[0]->m_queue.pop();
                auto out_1 = output_tensors.outputs[1]->m_queue.pop();
                auto out_2 = output_tensors.outputs[2]->m_queue.pop();
                auto raw_input = input_tensor.m_queue.pop();
                output_callback_ctr[0]--;
                output_callback_ctr[1]--;
                output_callback_ctr[2]--;
                lock.unlock();
                if ( !out_0 || !out_1 || !out_2 || !raw_input) {
                    std::cerr << "Failed to pop input or output buffer" << std::endl;
                    pp_status = HAILO_INTERNAL_FAILURE;
                    return;
                }
                if (print) {
                    std::unique_lock<std::mutex> lock(print_mutex);
                    std::cout << "post-process async write " << pp_ctr << std::endl;
                }
                cv::Mat raw_frame(camera->getHeight(), camera->getWidth(), CV_8UC3, static_cast<void*>(raw_input.get()));

                FeatureMap feature_map_2(out_2, output_tensors.outputs[2]->m_height, output_tensors.outputs[2]->m_width, output_tensors.outputs[2]->m_channels, 
                default_anchors_num, default_feature_map_channels, output_tensors.outputs[2]->m_qp_zp, output_tensors.outputs[2]->m_qp_scale, default_conf_threshold, {116, 90, 156, 198, 373, 326});
                FeatureMap feature_map_1(out_1, output_tensors.outputs[1]->m_height, output_tensors.outputs[1]->m_width, output_tensors.outputs[1]->m_channels, 
                default_anchors_num, default_feature_map_channels, output_tensors.outputs[1]->m_qp_zp, output_tensors.outputs[1]->m_qp_scale, default_conf_threshold, {30, 61, 62, 45, 59, 119});
                FeatureMap feature_map_0(out_0, output_tensors.outputs[0]->m_height, output_tensors.outputs[0]->m_width, output_tensors.outputs[0]->m_channels, 
                default_anchors_num, default_feature_map_channels, output_tensors.outputs[0]->m_qp_zp, output_tensors.outputs[0]->m_qp_scale, default_conf_threshold, {10, 13, 16, 30, 33, 23});
                
                YoloPost yolo_post;
                yolo_post.feature_maps.push_back(feature_map_2);
                yolo_post.feature_maps.push_back(feature_map_1);
                yolo_post.feature_maps.push_back(feature_map_0);

                std::vector<DetectionObject> detections = yolo_post.decode();
                // // -------------------------------------------------------------------------------------------------------------------

                for (auto& detection : detections) {
                    if (detection.confidence > 0) {
                        if (print) {
                            std::unique_lock<std::mutex> lock(print_mutex);
                            std::cout << "detection: id: "  << detection.class_id << ", bbox: " << detection.xmin << ", " << detection.ymin << ", " << detection.xmax << ", " << detection.ymax << ", " << detection.confidence << std::endl;
                        }
                        cv::rectangle(raw_frame, cv::Point2f(detection.xmin, detection.ymin), 
                            cv::Point2f(detection.xmax, detection.ymax), 
                            cv::Scalar(0, 0, 255), 1);
                    }
                }
#ifdef SAVE_TO_FILE
                video << raw_frame; // add in order to save to file the processed video
#endif
    
                pp_ctr++;
            }
#ifdef SAVE_TO_FILE
            video.release(); // add in order to save to file the processed video
#endif
            pp_status = HAILO_SUCCESS;
        });
        // ------------------------------------------------ join threads ----------------------------------------------------------------
        input_thread.join();
        for (auto& output_thread : output_threads) {
            output_thread.join();
        }
        pp_thread.join();
        std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
        std::cout << "FPS = " << get_num_frames_processed() * 1000 / std::chrono::duration_cast<std::chrono::milliseconds> (end_time - begin_time).count() << std::endl;
        // ----------------------------------------------- check statuses --------------------------------------------------------------

        if ((HAILO_STREAM_NOT_ACTIVATED != input_status) && (HAILO_SUCCESS != input_status)) {
            std::cerr << "Got unexpected status: " << input_status << " from input thread" << std::endl;
            return input_status;
        }
        if (std::any_of(output_statuses.begin(), output_statuses.end(),
                     [](const std::atomic<hailo_status>& status) {
                         return (status != HAILO_STREAM_NOT_ACTIVATED) && (status != HAILO_SUCCESS);
                     })) {
            std::cerr << "Got unexpected status from output thread" << std::endl;
            return HAILO_INTERNAL_FAILURE; // TODO: return specific error code
        }
        if ((HAILO_STREAM_NOT_ACTIVATED != pp_status) && (HAILO_SUCCESS != pp_status)) {
            std::cerr << "Got unexpected status: " << pp_status << " from post-process thread" << std::endl;
            return pp_status;
        }

        std::cout << "Inference finished successfully" << std::endl;
        return HAILO_SUCCESS;
    }
};

int main() {
    // -------------------------------------------- params -----------------------------------------------------------------------
#ifdef _EMBEDDED_
    const std::string video_source = "640.jpg";
    const bool print = true;
    const std::string hef_path = "yolov5m_h15.hef";
#else
    const std::string video_source = "640.mp4";
    const bool print = true;
    const std::string hef_path = "yolov5m_wo_spp_60p.hef";
#endif
    // -------------------------------------------- main -------------------------------------------------------------------------
    App app(video_source);
    hailo_status status = app.init(hef_path, print);
    if (status != HAILO_SUCCESS) {
        std::cerr << "Failed to init app, error: " << status << std::endl;
        return status;
    }
    status = app.run();
    if (status != HAILO_SUCCESS) {
        std::cerr << "Failed while running, error: " << status << std::endl;
        return status;
    }
    return 0;
}
