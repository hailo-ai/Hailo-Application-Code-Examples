#include "hailo/hailort.hpp"
#include "common.h"
#include "yolo_post_processing.hpp"

#include <iostream>
#include <chrono>
#include <mutex>
#include <future>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>


constexpr bool QUANTIZED = true;
constexpr hailo_format_type_t FORMAT_TYPE = HAILO_FORMAT_TYPE_AUTO;

using namespace hailort;

void print_inference_statistics(std::chrono::duration<double> inference_time,
                                std::chrono::duration<double> postprocess_time, 
                                std::string hef_file, double frame_count){
    std::cout << BOLDGREEN << "\n-I-----------------------------------------------" << std::endl;
    std::cout << "-I- " << hef_file.substr(0, hef_file.find(".")) << std::endl;
    std::cout << "-I-----------------------------------------------" << std::endl;
    std::cout << "\n-I-----------------------------------------------" << std::endl;
    std::cout << "-I- Inference                                    " << std::endl;
    std::cout << "-I-----------------------------------------------" << std::endl;
    std::cout << "-I- Total time:   " << inference_time.count() << " sec" << std::endl;
    std::cout << "-I- Average FPS:  " << frame_count / (inference_time.count()) << std::endl;
    std::cout << "-I- Latency:      " << 1.0 / (frame_count / (inference_time.count()) / 1000) << " ms" << std::endl;
    std::cout << "-I-----------------------------------------------" << std::endl;
    std::cout << "\n-I-----------------------------------------------" << std::endl;
    std::cout << "-I- Postprocess                                    " << std::endl;
    std::cout << "-I-----------------------------------------------" << std::endl;
    std::cout << "-I- Total time:   " << postprocess_time.count() << " sec" << std::endl;
    std::cout << "-I- Average FPS:  " << frame_count / (postprocess_time.count()) << std::endl;
    std::cout << "-I- Latency:      " << 1.0 / (frame_count / (postprocess_time.count()) / 1000) << " ms" << std::endl;
    std::cout << "-I-----------------------------------------------" << std::endl << RESET;
}

inline bool ends_with(std::string const & value, std::string const & ending) {
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}


hailo_status post_processing_all(std::vector<std::shared_ptr<FeatureData>> &features, 
                                std::chrono::duration<double>& postprocess_time, cv::Mat& frame, std::string arch,
                                float32_t* detections, int max_num_detections, float thr)
{
    auto status = HAILO_SUCCESS;
    std::sort(features.begin(), features.end(), &FeatureData::sort_tensors_by_size);
    std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::high_resolution_clock::now();

    auto detections_struct = post_processing(max_num_detections, thr, arch,
            features[0]->m_buffers.get_read_buffer().data(), features[0]->m_qp_zp, features[0]->m_qp_scale,
            features[1]->m_buffers.get_read_buffer().data(), features[1]->m_qp_zp, features[1]->m_qp_scale,
            features[2]->m_buffers.get_read_buffer().data(), features[2]->m_qp_zp, features[2]->m_qp_scale);
    
    for (auto &feature : features) {
        feature->m_buffers.release_read_buffer();
    }

    int num_detections = 0;
    int bytes_in_float = 6;
    size_t detections_4_byte_idx = 0;
    for (auto& detection : detections_struct) {
        if (detection.confidence >= thr && detections_4_byte_idx < max_num_detections * bytes_in_float) { 
 
            detections[detections_4_byte_idx++] = detection.ymin;
            detections[detections_4_byte_idx++] = detection.xmin;
            detections[detections_4_byte_idx++] = detection.ymax;
            detections[detections_4_byte_idx++] = detection.xmax;
            detections[detections_4_byte_idx++] = detection.confidence;
            detections[detections_4_byte_idx++] = static_cast<float32_t>(detection.class_id);

            num_detections++;
        }
    }

    std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::high_resolution_clock::now();
    postprocess_time = t_end - t_start;

    return status;
}


hailo_status read_all(OutputVStream& output_vstream, std::shared_ptr<FeatureData> feature,
                    std::chrono::time_point<std::chrono::system_clock>& read_time_vec) { 

    auto& buffer = feature->m_buffers.get_write_buffer();
    hailo_status status = output_vstream.read(MemoryView(buffer.data(), buffer.size()));
    feature->m_buffers.release_write_buffer();
    if (HAILO_SUCCESS != status) {
        std::cerr << "Failed reading with status = " <<  status << std::endl;
        return status;
    }

    read_time_vec = std::chrono::high_resolution_clock::now();
    return HAILO_SUCCESS;
}

hailo_status write_all(InputVStream& input_vstream, std::string image_path, 
                        std::chrono::time_point<std::chrono::system_clock>& write_time_vec, cv::Mat& frame) {

    hailo_status status = HAILO_SUCCESS;
    
    auto input_shape = input_vstream.get_info().shape;
    int height = input_shape.height;
    int width = input_shape.width;

    write_time_vec = std::chrono::high_resolution_clock::now();
    if (ends_with(image_path, ".jpg") || ends_with(image_path, ".png") || ends_with(image_path, ".jpeg")) {
        frame = cv::imread(image_path,  cv::IMREAD_COLOR);
        if (frame.channels() == 3) {
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        }
        if (frame.rows != height || frame.cols != width) {
            cv::resize(frame, frame, cv::Size(width, height), cv::INTER_AREA);
        }
        status = input_vstream.write(MemoryView(frame.data, input_vstream.get_frame_size()));
        if (HAILO_SUCCESS != status) {
            return status;
        }
    }
    status = HAILO_SUCCESS;
    return status;
}


hailo_status create_feature(hailo_vstream_info_t vstream_info, size_t output_frame_size, std::shared_ptr<FeatureData> &feature) {
    feature = std::make_shared<FeatureData>(static_cast<uint32_t>(output_frame_size), vstream_info.quant_info.qp_zp,
        vstream_info.quant_info.qp_scale, vstream_info.shape.width);

    return HAILO_SUCCESS;
}


hailo_status run_inference(std::vector<InputVStream>& input_vstream, std::vector<OutputVStream>& output_vstreams, std::string image_path,
                    std::chrono::time_point<std::chrono::system_clock>& write_time_vec,
                    std::vector<std::chrono::time_point<std::chrono::system_clock>>& read_time_vec,
                    std::chrono::duration<double>& inference_time, std::chrono::duration<double>& postprocess_time, std::string arch,
                    float32_t* detections, int max_num_detections) {

    hailo_status status = HAILO_UNINITIALIZED;
    
    auto output_vstreams_size = output_vstreams.size();

    std::vector<std::shared_ptr<FeatureData>> features;
    features.reserve(output_vstreams_size);
    for (size_t i = 0; i < output_vstreams_size; i++) {
        std::shared_ptr<FeatureData> feature(nullptr);
        auto status = create_feature(output_vstreams[i].get_info(), output_vstreams[i].get_frame_size(), feature);
        if (HAILO_SUCCESS != status) {
            std::cerr << "Failed creating feature with status = " << status << std::endl;
            return status;
        }

        features.emplace_back(feature);
    }

    cv::Mat frame;
    auto input_thread(std::async(write_all, std::ref(input_vstream[0]), image_path, std::ref(write_time_vec), std::ref(frame)));

    // Create read threads
    std::vector<std::future<hailo_status>> output_threads;
    output_threads.reserve(output_vstreams_size);
    for (size_t i = 0; i < output_vstreams_size; i++) {
        output_threads.emplace_back(std::async(read_all, std::ref(output_vstreams[i]), features[i], std::ref(read_time_vec[i])));
    }

    float thr = 0.5f;
    auto pp_thread(std::async(post_processing_all, std::ref(features), std::ref(postprocess_time), std::ref(frame), arch, detections, max_num_detections, thr));

    for (size_t i = 0; i < output_threads.size(); i++) {
        status = output_threads[i].get();
    }
    auto input_status = input_thread.get();
    auto pp_status = pp_thread.get();

    if (HAILO_SUCCESS != input_status) {
        std::cerr << "Write thread failed with status " << input_status << std::endl;
        return input_status; 
    }
    if (HAILO_SUCCESS != status) {
        std::cerr << "Read failed with status " << status << std::endl;
        return status;
    }
    if (HAILO_SUCCESS != pp_status) {
        std::cerr << "Post-processing failed with status " << pp_status << std::endl;
        return pp_status;
    }

    inference_time = read_time_vec[0] - write_time_vec;
    for (size_t i = 1; i < output_vstreams.size(); i++){
        if (inference_time.count() < (double)(read_time_vec[i] - write_time_vec).count())
            inference_time = read_time_vec[i] - write_time_vec;
    }

    status = HAILO_SUCCESS;
    return status;
}

void print_net_banner(std::pair<std::vector<hailort::InputVStream>, std::vector<hailort::OutputVStream>> &vstreams) {
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
    std::cout << BOLDMAGENTA << "-I-  Network  Name                                     " << std::endl << RESET;
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
    for (auto const& value: vstreams.first) {
        std::cout << MAGENTA << "-I-  IN:  " << value.name() <<std::endl << RESET;
    }
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
    for (auto const& value: vstreams.second) {
        std::cout << MAGENTA << "-I-  OUT: " << value.name() <<std::endl << RESET;
    }
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------\n" << std::endl << RESET;
}

Expected<std::shared_ptr<ConfiguredNetworkGroup>> configure_network_group(VDevice &vdevice, std::string yolov_hef)
{
    auto hef_exp = Hef::create(yolov_hef);
    if (!hef_exp) {
        return make_unexpected(hef_exp.status());
    }
    auto hef = hef_exp.release();

    auto configure_params = hef.create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
    if (!configure_params) {
        return make_unexpected(configure_params.status());
    }

    auto network_groups = vdevice.configure(hef, configure_params.value());
    if (!network_groups) {
        return make_unexpected(network_groups.status());
    }

    if (1 != network_groups->size()) {
        std::cerr << "Invalid amount of network groups" << std::endl;
        return make_unexpected(HAILO_INTERNAL_FAILURE);
    }

    return std::move(network_groups->at(0));
}

extern "C" int infer_wrapper(const char* hef_path, const char* image_path, const char* arch, float* detections, int max_num_detections) {

    hailo_status status = HAILO_UNINITIALIZED;

    std::chrono::duration<double> total_time;
    std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::high_resolution_clock::now();

    std::chrono::time_point<std::chrono::system_clock> write_time_vec;
    std::chrono::duration<double> inference_time;
    std::chrono::duration<double> postprocess_time;

    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) {
        std::cerr << "Failed create vdevice, status = " << vdevice_exp.status() << std::endl;
        return vdevice_exp.status();
    }
    auto vdevice = vdevice_exp.release();

    auto network_group_exp = configure_network_group(*vdevice, hef_path);
    if (!network_group_exp) {
        std::cerr << "Failed to configure network group " << hef_path << std::endl;
        return network_group_exp.status();
    }
    auto network_group = network_group_exp.release();

    auto vstreams_exp = VStreamsBuilder::create_vstreams(*network_group, QUANTIZED, FORMAT_TYPE);
    if (!vstreams_exp) {
        std::cerr << "Failed creating vstreams " << vstreams_exp.status() << std::endl;
        return vstreams_exp.status();
    }
    auto vstreams = vstreams_exp.release();

    std::vector<std::chrono::time_point<std::chrono::system_clock>> read_time_vec(vstreams.second.size());

    print_net_banner(vstreams);

    status = run_inference(std::ref(vstreams.first), 
                        std::ref(vstreams.second), 
                        image_path, write_time_vec, read_time_vec, 
                        inference_time, postprocess_time, arch,
                        detections, max_num_detections);

    if (HAILO_SUCCESS != status) {
        std::cerr << "Failed running inference with status = " << status << std::endl;
        return status;
    }

    print_inference_statistics(inference_time, postprocess_time, hef_path, 1.f);

    std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::high_resolution_clock::now();
    total_time = t_end - t_start;

    std::cout << BOLDBLUE << "\n-I- Inference run finished successfully" << RESET << std::endl;
    return HAILO_SUCCESS;
}
