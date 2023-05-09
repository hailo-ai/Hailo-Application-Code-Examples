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
std::mutex m;

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
    std::cout << YELLOW << "\n-I- Starting postprocessing\n" << std::endl << RESET;

    auto detections_struct = post_processing(max_num_detections, thr, arch,
            features[0]->m_buffers.get_read_buffer().data(), features[0]->m_qp_zp, features[0]->m_qp_scale,
            features[1]->m_buffers.get_read_buffer().data(), features[1]->m_qp_zp, features[1]->m_qp_scale,
            features[2]->m_buffers.get_read_buffer().data(), features[2]->m_qp_zp, features[2]->m_qp_scale);
    
    for (auto &feature : features) {
        feature->m_buffers.release_read_buffer();
    }

    size_t detections_4_byte_idx = 0;
    for (auto& detection : detections_struct) {
        if (detection.confidence >= thr && detections_4_byte_idx < max_num_detections*6) {  // 6 floats that represent 1 detection
            // std::cout << "Detection: " << get_coco_name_from_int(detection.class_id) << ", Confidence: " << std::fixed << std::setprecision(2) << detection.confidence * 100.0 << "%" << std::endl;
            
            // heavy copy :(
            detections[detections_4_byte_idx++] = detection.ymin;
            detections[detections_4_byte_idx++] = detection.xmin;
            detections[detections_4_byte_idx++] = detection.ymax;
            detections[detections_4_byte_idx++] = detection.xmax;
            detections[detections_4_byte_idx++] = detection.confidence;
            detections[detections_4_byte_idx++] = static_cast<float32_t>(detection.class_id);
        }
    }
    frame.release();

    std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::high_resolution_clock::now();
    postprocess_time = t_end - t_start;

    return status;
}


hailo_status read_all(OutputVStream& output_vstream, std::shared_ptr<FeatureData> feature,
                    std::chrono::time_point<std::chrono::system_clock>& read_time_vec) { 

    m.lock();
    std::cout << GREEN << "-I- Started read thread: " << info_to_str(output_vstream.get_info()) << std::endl << RESET;
    m.unlock(); 

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
    m.lock();
    std::cout << CYAN << "-I- Started write thread: " << info_to_str(input_vstream.get_info()) << std::endl << RESET;
    m.unlock();

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
            std::cout << " resizing........." << std::endl;
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

    cv::Mat frame; // cv::Mat frame((int));
    auto input_thread(std::async(write_all, std::ref(input_vstream[0]), image_path, std::ref(write_time_vec), std::ref(frame)));

    // Create read threads
    std::vector<std::future<hailo_status>> output_threads;
    output_threads.reserve(output_vstreams_size);
    for (size_t i = 0; i < output_vstreams_size; i++) {
        output_threads.emplace_back(std::async(read_all, std::ref(output_vstreams[i]), features[i], std::ref(read_time_vec[i])));
    }

    float thr = 0.3f;
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

    std::cout << BOLDBLUE << "\n-I- Inference finished successfully" << RESET << std::endl;

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


Expected<std::vector<std::pair<std::vector<InputVStream>, std::vector<OutputVStream>>>> build_vstreams(
    const std::vector<std::shared_ptr<ConfiguredNetworkGroup>> &configured_network_groups) {
    std::vector<std::pair<std::vector<InputVStream>, std::vector<OutputVStream>>> vstreams_per_network_group;

    for (auto &network_group : configured_network_groups) {
        auto vstreams_exp = VStreamsBuilder::create_vstreams(*network_group, QUANTIZED, FORMAT_TYPE);
        if (!vstreams_exp) {
            return make_unexpected(vstreams_exp.status());
        }
        vstreams_per_network_group.emplace_back(vstreams_exp.release());
    }
    return vstreams_per_network_group;
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

    auto input_vstream_params = network_group->make_input_vstream_params(false, HAILO_FORMAT_TYPE_UINT8, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    auto output_vstream_params = network_group->make_output_vstream_params(false, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE); // HAILO_FORMAT_TYPE_UINT16
    auto input_vstreams  = VStreamsBuilder::create_input_vstreams(*network_group, input_vstream_params.value());
    auto output_vstreams = VStreamsBuilder::create_output_vstreams(*network_group, output_vstream_params.value());
    if (!input_vstreams or !output_vstreams) {
        std::cerr << "-E- Failed creating input: " << input_vstreams.status() << " output status:" << output_vstreams.status() << std::endl;
        return input_vstreams.status();
    }
    auto vstreams = std::make_pair(input_vstreams.release(), output_vstreams.release());

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

    std::cout << BOLDBLUE << "\n-I- Application run finished successfully" << RESET << std::endl;
    std::cout << BOLDBLUE << "-I- Total application run time: " << (double)total_time.count() << " sec" << RESET << std::endl;
    return HAILO_SUCCESS;
}

// int main() {

//     std::string yolov_hef = "/home/batshevak/projects/new_jj/Hailo-Application-Code-Examples/infer_wrapper/infer_wrapper/yolov5m_wo_spp_60p.hef";
//     std::string image_path = "/home/batshevak/projects/new_jj/Hailo-Application-Code-Examples/infer_wrapper/infer_wrapper/images/zidane_640.jpg";
//     std::string arch = "yolov5";
//     // const int FLOAT = 4;
//     const int NUM_DETECTIONS = 20;
//     const int SIZE_DETECTION = 6;
//     size_t detections_size = NUM_DETECTIONS * SIZE_DETECTION;
//     std::vector<float32_t> detections(detections_size);
//     int max_num_detections = NUM_DETECTIONS;
//     return infer_wrapper(yolov_hef, image_path, arch, std::ref(detections), max_num_detections);
// }
