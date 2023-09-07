
#include "hailo/hailort.hpp"
#include "tensors_buffers.hpp"
#include "yolo_post.hpp"

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

std::atomic<int> read_async_counter(0);
std::atomic<int> write_async_counter(0);
constexpr auto TIMEOUT = std::chrono::milliseconds(1000);

class VideoCaptureWrapper {
private:
    cv::VideoCapture capture;
    size_t frame_size;
    int width;
    int height;

public:
    VideoCaptureWrapper(int deviceIndex) : capture(deviceIndex), frame_size(0) {
        if (!capture.isOpened()) {
            std::cerr << "Error: Could not open camera device" << std::endl;
            exit(1);
        }
    }

    VideoCaptureWrapper(const std::string& filename) : capture(filename) {
        if (!capture.isOpened()) {
            std::cerr << "Error: Could not open video file" << std::endl;
            exit(1);
        }
    }

    int getNextFrame(AlignedBuffer buffer) {
        cv::Mat frame(height, width, CV_8UC3, static_cast<void*>(buffer.get()));
        capture >> frame;
        if (frame.empty()) {
            // std::cerr << "frame is empty, meaning we finished reading all frames" << std::endl;
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    bool isOpened() const {
        return capture.isOpened();
    }

    hailo_status setFrameSize(size_t frame_size_hef) {
        size_t frame_size_capture = width * height * 3; // TODO: const
        if (frame_size_hef != frame_size_capture) {
            std::cerr << "Error: Frame size of HEF and capture device do not match, HEF: " << frame_size_hef << ", capture: " << frame_size_capture << std::endl;
            return HAILO_INVALID_ARGUMENT;
        }
        frame_size = frame_size_hef;
        return HAILO_SUCCESS;
    }

    hailo_status setHeightWidth(double height_hef, double width_hef) {
        double height_capture = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
        double width_capture = capture.get(cv::CAP_PROP_FRAME_WIDTH);
        if (height_hef != height_capture || width_hef != width_capture) {
            std::cerr << "Error: Frame size of HEF and capture device do not match, hxw HEF: " << height_hef << "x" << width_hef << ", hxw capture: " << height_capture << "x" << width_capture << std::endl;
            return HAILO_INVALID_ARGUMENT; // TODO: maybe resize video and not fail, and only print warning
        }
        height = height_hef;
        width = width_hef;
        return HAILO_SUCCESS;
    }

    int getWidth() {
        return width;
    }

    int getHeight() {
        return height;
    }
};

template <class T>
bool compare_queue_size(const T& a, const T& b) {
    return a.get().get_async_max_queue_size().value() < b.get().get_async_max_queue_size().value(); // b7: need to check expected bla bla
}

// // TODO: find a better place to decalre this, maybe inside a class? but may make some errors in function signature
// static void input_async_callback(const InputStream::CompletionInfo &completion_info)
// {
//     // Real applications can free the buffer or reuse it for next transfer.
//     if ((HAILO_SUCCESS != completion_info.status) && (HAILO_STREAM_ABORTED_BY_USER  != completion_info.status)) {
//         // We will get HAILO_STREAM_ABORTED_BY_USER  when activated_network_group is destructed.
//         std::cerr << "Got an unexpected status on callback. status=" << completion_info.status << std::endl;
//     }
// }

// static void output_async_callback(const OutputStream::CompletionInfo &completion_info)
// {
//     // Real applications can free the buffer or forward it to post-process/display.
//     if ((HAILO_SUCCESS != completion_info.status) && (HAILO_STREAM_ABORTED_BY_USER != completion_info.status)) {
//         // We will get HAILO_STREAM_ABORTED_BY_USER when activated_network_group is destructed.
//         std::cerr << "Got an unexpected status on callback. status=" << completion_info.status << std::endl;
//     }
// }

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// std::mutex output_callback_mutex;
// std::condition_variable output_callback_cv;
// std::atomic<int> output_callback_notifications(0);

// OutputStream::TransferDoneCallback output_async_callback = [](const OutputStream::CompletionInfo &completion_info) {
//         hailo_status status = HAILO_UNINITIALIZED;
//         switch (completion_info.status) {
//         case HAILO_SUCCESS:
//             // Real applications can forward the buffer to post-process/display.
//             if ((HAILO_SUCCESS != status) && (HAILO_STREAM_NOT_ACTIVATED != status)) { // b7: why we need to check here?? why the default isn't enough? 
//             // +  what's the difference between HAILO_STREAM_NOT_ACTIVATED & HAILO_STREAM_ABORTED_BY_USER? 
//             // why the stream may be not activated? (is it related to the network group?)
//                 std::cerr << "Failed read async with status=" << status << std::endl;
//             }
//             break;
//         case HAILO_STREAM_ABORTED_BY_USER:
//             // Transfer was canceled, finish gracefully.
//             break;
//         default:
//             std::cerr << "Got an unexpected status on callback. status=" << completion_info.status << std::endl;
//         }
//     };
//     InputStream::TransferDoneCallback input_async_callback = [](const InputStream::CompletionInfo &completion_info) {
//         hailo_status status = HAILO_UNINITIALIZED;
//         switch (completion_info.status) {
//         case HAILO_SUCCESS:
//             // Real applications may free the buffer and replace it with new buffer ready to be sent.
//             if ((HAILO_SUCCESS != status) && (HAILO_STREAM_NOT_ACTIVATED != status)) {
//                 std::cerr << "Failed read async with status=" << status << std::endl;
//             }
//             break;
//         case HAILO_STREAM_ABORTED_BY_USER:
//             // Transfer was canceled, finish gracefully.
//             break;
//         default:
//             std::cerr << "Got an unexpected status on callback. status=" << completion_info.status << std::endl;
//         }
//     };
///////////////////////////////////////////////////////////////////////////////////////////////////////////

class App {
private:
    VideoCaptureWrapper camera;
    std::unique_ptr<hailort::Device> device;
    std::shared_ptr<ConfiguredNetworkGroup> network_group; // all the .get() because of reference_wrapper are ugly, maybe save as reference of inputs & outputs vstreams instead (or in addition)
    int num_outputs;
    std::mutex print_mutex;
    std::mutex app_mutex;
    std::condition_variable app_cv;
    int app_notifications;
    int input_ctr; // TODO: maybe it should be atomic?
    std::vector<int> output_ctr;
    std::vector<int> output_callback_ctr;
    int pp_ctr;
    bool continue_run;

public:
    App(int cameraIndex) : camera(cameraIndex), num_outputs(0), app_notifications(0), input_ctr(0), output_ctr(0), output_callback_ctr(0), pp_ctr(0), continue_run(false) {}
    App(const std::string source) : camera(source), num_outputs(0), app_notifications(0), input_ctr(0), output_ctr(0), output_callback_ctr(0), pp_ctr(0), continue_run(false) {}

    hailo_status init(const std::string& hef_path) {
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
        auto status = camera.setHeightWidth(network_group->get_input_streams()[0].get().get_info().hw_shape.height, network_group->get_input_streams()[0].get().get_info().hw_shape.width);
        if (status != HAILO_SUCCESS) {
            std::cerr << "Failed to set input width and height" << std::endl;
            return status;
        }
        status = camera.setFrameSize(network_group->get_input_streams()[0].get().get_frame_size()); // assuming only 1 input //TODO: change duplicate: setHeightWidth & setFrameSize
        if (status != HAILO_SUCCESS) {
            std::cerr << "Failed to set input frame size" << std::endl;
            return status;
        }
        num_outputs = network_group->get_output_vstream_infos()->size();
        output_ctr.resize(num_outputs, 0);
        output_callback_ctr.resize(num_outputs, 0);

        continue_run = true;
        // TODO: add num_inputs to support multiple inputs
        return HAILO_SUCCESS;
    }

    VideoCaptureWrapper& getCamera() { // TODO: remove, debug only
        return camera;
    }    

    hailort::Expected<std::shared_ptr<hailort::ConfiguredNetworkGroup>> configureNetwork(const std::string& hef_path) {
        auto hef_exp = Hef::create(hef_path);
        if (!hef_exp) {
            return make_unexpected(hef_exp.status());
        }
        auto hef = hef_exp.release(); // may make errors since the hef memory will be freed?

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

    // TODO: find a better place to decalre this, maybe inside a class? but may make some errors in function signature
    static void input_async_callback(const InputStream::CompletionInfo &completion_info) // b7: why it has to be static?
    {
        // Real applications can free the buffer or reuse it for next transfer.
        if ((HAILO_SUCCESS != completion_info.status) && (HAILO_STREAM_ABORTED_BY_USER  != completion_info.status)) {
            // We will get HAILO_STREAM_ABORTED_BY_USER  when activated_network_group is destructed.
            std::cerr << "Got an unexpected status on callback. status=" << completion_info.status << std::endl;
        }
    }

    static void output_async_callback(const OutputStream::CompletionInfo &completion_info)
    {
        // Real applications can free the buffer or forward it to post-process/display.
        if ((HAILO_SUCCESS != completion_info.status) && (HAILO_STREAM_ABORTED_BY_USER != completion_info.status)) {
            // We will get HAILO_STREAM_ABORTED_BY_USER when activated_network_group is destructed.
            std::cerr << "Got an unexpected status on callback. status=" << completion_info.status << std::endl;
        }
        // std::unique_lock<std::mutex> lock(app_mutex);
        // std::mutex app_mutex;
        // std::condition_variable app_cv;
        // int app_notifications;
    }

    hailo_status run() {
        // cv::namedWindow("Video Stream", cv::WINDOW_NORMAL);
        auto inputs = network_group->get_input_streams();
        auto outputs = network_group->get_output_streams();
        // --------------------------------------------------- create buffers for input / output -----------------------------------
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
        // --------------------------------------------------- input thread -------------------------------------------------------
        std::atomic<hailo_status> input_status(HAILO_UNINITIALIZED);
        std::thread input_thread([&print_mutex=print_mutex, &input_status, &inputs, &camera=camera, &input_tensor, &input_ctr=input_ctr, &continue_run=continue_run]() {
            // int input_ctr = 0;
            while (continue_run) {
                std::unique_lock<std::mutex> lock(print_mutex);
                lock.unlock();
                input_status = inputs[0].get().wait_for_async_ready(inputs[0].get().get_frame_size(), TIMEOUT);
                if (HAILO_SUCCESS != input_status) { return; }
                auto input_buffer = page_aligned_alloc(inputs[0].get().get_frame_size());
                auto get_frame_status = camera.getNextFrame(input_buffer);
                if (EXIT_SUCCESS != get_frame_status) {
                    continue_run = false;
                    return; 
                } // TODO: notify to read_all & post_process threads to finish their work and then return
                input_tensor.m_queue.push(input_buffer);
                input_status = inputs[0].get().write_async(input_buffer.get(), inputs[0].get().get_frame_size(), input_async_callback);
                if (HAILO_SUCCESS != input_status) { return; }
                if (input_ctr < 2 || input_ctr > 1230) {
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
        for (int i = 0; i < num_outputs; i++) { // the thread get all by reference, including i, that's why I think we get a segfault
            output_threads.emplace_back(std::thread([&print_mutex=print_mutex, &output_statuses, &outputs, i, &output_tensors, &input_ctr=input_ctr, &output_ctr=output_ctr[i], &continue_run=continue_run]() { // TODO: is output_ctr[i] ok???
            // int output_ctr = 0;
            while (output_ctr < input_ctr || continue_run) { // we haven't finished to process all the input frames // TODO: make sure it's < and not <=. Also if possible without bool continue_run, it will be clearer.
                std::unique_lock<std::mutex> lock(print_mutex);
                lock.unlock();
                output_statuses[i] = outputs[i].get().wait_for_async_ready(outputs[i].get().get_frame_size(), TIMEOUT);
                if (HAILO_SUCCESS != output_statuses[i]) { return; }

                auto output_buffer = page_aligned_alloc(outputs[i].get().get_frame_size());
                output_statuses[i] = outputs[i].get().read_async(output_buffer.get(), outputs[i].get().get_frame_size(), output_async_callback);
                if (HAILO_SUCCESS != output_statuses[i]) { return; }
                output_tensors.outputs[i].get()->m_queue.push(output_buffer);
                if (output_ctr < 2 || output_ctr > 1230) {
                    std::cout << "output async read " << output_ctr << ", thread " << i << std::endl;
                }
                output_ctr++;
                }
            }));
        }
        // --------------------------------------------------- post-process thread -------------------------------------------------------
        std::atomic<hailo_status> pp_status(HAILO_UNINITIALIZED);
        std::thread pp_thread([&print_mutex=print_mutex, &camera=camera, &input_tensor, &output_tensors, &input_ctr=input_ctr, &output_callback_ctr=output_callback_ctr, &pp_ctr=pp_ctr, &continue_run=continue_run]() {
            // int pp_ctr = 0;
            cv::VideoWriter video("./processed_video.mp4", cv::VideoWriter::fourcc('m','p','4','v'),30, cv::Size(camera.getWidth(), camera.getHeight()));
            while (pp_ctr < input_ctr || continue_run) { // we haven't finished to process all the input frames
                std::unique_lock<std::mutex> lock(print_mutex);
                lock.unlock();
                
                // we have to make sure that all 3 outputs finished calback before we can pop them
                // TODO: check with counter from callback for 3 out !!!
                auto out_0 = output_tensors.outputs[0]->m_queue.pop();
                auto out_1 = output_tensors.outputs[1]->m_queue.pop();
                auto out_2 = output_tensors.outputs[2]->m_queue.pop();
                auto raw_input = input_tensor.m_queue.pop();
                if ( !out_0 || !out_1 || !out_2 || !raw_input) {
                    std::cerr << "Failed to pop inut or output buffer" << std::endl;
                    return;
                }
                if (pp_ctr < 2 || pp_ctr > 1230) {
                    std::cout << "post-process async write " << pp_ctr << std::endl;
                }
                cv::Mat raw_frame(camera.getHeight(), camera.getWidth(), CV_8UC3, static_cast<void*>(raw_input.get()));

                FeatureMap feature_map_0(out_0, output_tensors.outputs[0]->m_height, output_tensors.outputs[0]->m_width, output_tensors.outputs[0]->m_channels, 
                default_anchors_num, default_feature_map_channels, output_tensors.outputs[0]->m_qp_zp, output_tensors.outputs[0]->m_qp_scale, default_conf_threshold, {116, 90, 156, 198, 373, 326});
                FeatureMap feature_map_1(out_1, output_tensors.outputs[1]->m_height, output_tensors.outputs[1]->m_width, output_tensors.outputs[1]->m_channels, 
                default_anchors_num, default_feature_map_channels, output_tensors.outputs[1]->m_qp_zp, output_tensors.outputs[1]->m_qp_scale, default_conf_threshold, {30, 61, 62, 45, 59, 119});
                FeatureMap feature_map_2(out_2, output_tensors.outputs[2]->m_height, output_tensors.outputs[2]->m_width, output_tensors.outputs[2]->m_channels, 
                default_anchors_num, default_feature_map_channels, output_tensors.outputs[2]->m_qp_zp, output_tensors.outputs[2]->m_qp_scale, default_conf_threshold, {10, 13, 16, 30, 33, 23});
                
                YoloPost yolo_post;
                yolo_post.feature_maps.push_back(feature_map_0);
                yolo_post.feature_maps.push_back(feature_map_1);
                yolo_post.feature_maps.push_back(feature_map_2);

                std::vector<DetectionObject> detections = yolo_post.decode();

                for (auto& detection : detections) {
                    if (detection.confidence > 0) {
                        std::cout << "detection: id: "  << detection.class_id << ", bbox: " << detection.xmin << ", " << detection.ymin << ", " << detection.xmax << ", " << detection.ymax << ", " << detection.confidence << std::endl;
                        // cv::Rect rect(detection.xmin, detection.ymin, detection.xmax, detection.ymax);
                        // cv::rectangle(raw_frame, rect, cv::Scalar(0, 255, 0));
                        cv::rectangle(raw_frame, cv::Point2f(detection.xmin, detection.ymin), 
                            cv::Point2f(detection.xmax, detection.ymax), 
                            cv::Scalar(0, 0, 255), 1);
                    }
                }
                // cv::Rect centerRect(camera.getWidth()/4, camera.getHeight()/4, (camera.getWidth()*3)/4 - camera.getWidth()/4, (camera.getHeight()*3)/4 - camera.getHeight()/4);
                // cv::rectangle(raw_frame, centerRect, cv::Scalar(0, 0, 255));
                video << raw_frame;
    
                pp_ctr++;
            }
            video.release(); // TODO: cleaner get out from thread (we always miss the last frame)
        });
        // --------------------------------------------------------------------------------------------------------------------------
        // After all async operations are launched, the inference is running.
        // std::this_thread::sleep_for(std::chrono::seconds(2));
        std::unique_lock<std::mutex> lock(print_mutex);
        std::cout << "still fine 5" << std::endl;
        lock.unlock();

        input_thread.join();
        // Make it stop. We explicitly destruct activated_network_group to stop all async I/O.
        // activated_network_group->reset();
        for (auto& output_thread : output_threads) {
            output_thread.join();
        }
        pp_thread.join();

        if ((HAILO_STREAM_NOT_ACTIVATED != input_status) && (HAILO_SUCCESS != input_status)) {
            std::cerr << "[b7] Got unexpected status: " << input_status << " from thread" << std::endl; //  << output_status_0
            return input_status; // TODO: check all statuses
        }

        std::cout << "Inference finished successfully" << std::endl;
        return HAILO_SUCCESS; // TODO: if all statuses are success, then return success
        }
    };

int main() {
    // -------------------------------------------- params -----------------------------------------------------------------------
    // int cameraIndex = 0; // You might need to adjust the camera index
    const std::string video_source = "640.mp4";
    const std::string hef_path = "/home/batshevak/useful/models/yolov5m_wo_spp_60p.hef";
    // -------------------------------------------- main -------------------------------------------------------------------------
    App app(video_source);
    hailo_status status = app.init(hef_path);
    if (status != HAILO_SUCCESS) {
        std::cerr << "Failed to init app, error: " << status << std::endl;
        return status;
    }
    status = app.run();
    if (status != HAILO_SUCCESS) {
        std::cerr << "Failed to while running, error: " << status << std::endl;
        return status;
    }

    return 0;
}