#include "hailo/hailort.hpp"
#include "common.h"

#include "common/hailo_objects.hpp"
#include "yolov8seg_postprocess.hpp"

#include <iostream>
#include <chrono>
#include <mutex>
#include <future>
#include <random>

constexpr bool QUANTIZED = true;
constexpr hailo_format_type_t FORMAT_TYPE = HAILO_FORMAT_TYPE_AUTO;
std::mutex m;

using namespace hailort;

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


template <typename T>
hailo_status post_processing_all(std::vector<std::shared_ptr<FeatureData<T>>> &features, size_t frame_count, 
                                std::chrono::time_point<std::chrono::system_clock>& postprocess_time, std::vector<cv::Mat>& frames, 
                                double org_height, double org_width, bool nms_on_hailo, std::string model_type) {

    auto status = HAILO_SUCCESS;

    std::sort(features.begin(), features.end(), &FeatureData<T>::sort_tensors_by_size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> random_index(0, COLORS.size() - 1);

    // cv::VideoWriter video("./processed_video.mp4", cv::VideoWriter::fourcc('m','p','4','v'),30, cv::Size((int)org_width, (int)org_height));

    m.lock();
    std::cout << YELLOW << "\n-I- Starting postprocessing\n" << std::endl << RESET;
    m.unlock();

    for (size_t i = 0; i < frame_count; i++){
        HailoROIPtr roi = std::make_shared<HailoROI>(HailoROI(HailoBBox(0.0f, 0.0f, 1.0f, 1.0f)));
        
        for (uint j = 0; j < features.size(); j++) {
            roi->add_tensor(std::make_shared<HailoTensor>(reinterpret_cast<T*>(features[j]->m_buffers.get_read_buffer().data()), features[j]->m_vstream_info));
        }

        auto filtered_masks = filter(roi, (int)org_height, (int)org_width);
    
        for (auto &feature : features) {
            feature->m_buffers.release_read_buffer();
        }

        std::vector<HailoDetectionPtr> detections = hailo_common::get_hailo_detections(roi);
        cv::resize(frames[0], frames[0], cv::Size((int)org_width, (int)org_height), 1);

        for (auto &detection : detections) {
            if (detection->get_confidence()==0) {
                continue;
            }

            HailoBBox bbox = detection->get_bbox();
        
            cv::rectangle(frames[0], cv::Point2f(bbox.xmin() * float(org_width), bbox.ymin() * float(org_height)), 
                        cv::Point2f(bbox.xmax() * float(org_width), bbox.ymax() * float(org_height)), 
                        cv::Scalar(0, 0, 255), 1);
            
            std::cout << "Detection: " << detection->get_label() << ", Confidence: " << std::fixed << std::setprecision(2) << detection->get_confidence() * 100.0 << "%" << std::endl;
        }

        for (auto& mask : filtered_masks){
            cv::Mat overlay = cv::Mat(frames[0].rows, frames[0].cols, CV_8UC3, cv::Scalar(0));
            auto pixel_color = COLORS[random_index(gen)];
            for (int r = 0; r < mask.rows; r++) {
                for (int c = 0; c < mask.cols ; c++) {
                    if (mask.at<float>(r, c) > 0.7) {
                        overlay.at<cv::Vec3b>(r,c) = pixel_color;
                    }
                }
            }
            cv::addWeighted(frames[0], 1, overlay, 0.5, 0.0, frames[0]);
            overlay.release();
            mask.release();
        }

        // cv::imshow("Display window", frames[0]);
        // cv::waitKey(0);

        // video.write(frames[0]);
        // cv::imwrite("output_image.jpg", frames[0]);
        
        frames[0].release();
        
        m.lock();
        frames.erase(frames.begin());
        m.unlock();
    }
    postprocess_time = std::chrono::high_resolution_clock::now();
    // video.release();

    return status;
}

template <typename T>
hailo_status read_all(OutputVStream& output_vstream, std::shared_ptr<FeatureData<T>> feature, size_t frame_count) { 

    m.lock();
    std::cout << GREEN << "-I- Started read thread: " << info_to_str(output_vstream.get_info()) << std::endl << RESET;
    m.unlock(); 

    if (frame_count == -1){
        for (;;) {
            std::vector<T>& buffer = feature->m_buffers.get_write_buffer();
            hailo_status status = output_vstream.read(MemoryView(buffer.data(), buffer.size()));
            feature->m_buffers.release_write_buffer();
            if (HAILO_SUCCESS != status) {
                std::cerr << "Failed reading with status = " <<  status << std::endl;
                return status;
            }
        }
    }
    else {
        for (size_t i = 0; i < (size_t)frame_count; i++) {
            std::vector<T>& buffer = feature->m_buffers.get_write_buffer();
            hailo_status status = output_vstream.read(MemoryView(buffer.data(), buffer.size()));
            feature->m_buffers.release_write_buffer();
            if (HAILO_SUCCESS != status) {
                std::cerr << "Failed reading with status = " <<  status << std::endl;
                return status;
            }
        }
    }

    return HAILO_SUCCESS;
}

hailo_status use_single_frame(InputVStream& input_vstream, std::chrono::time_point<std::chrono::system_clock>& write_time_vec,
                                std::vector<cv::Mat>& frames, cv::Mat& image, int frame_count){

    hailo_status status = HAILO_SUCCESS;
    write_time_vec = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < frame_count; i++) {
        m.lock();
        frames.push_back(image);
        m.unlock();
        status = input_vstream.write(MemoryView(frames[frames.size() - 1].data, input_vstream.get_frame_size()));
        if (HAILO_SUCCESS != status)
            return status;
    }

    return HAILO_SUCCESS;
}


hailo_status write_all(InputVStream& input_vstream, std::string input_path, 
                        std::chrono::time_point<std::chrono::system_clock>& write_time_vec, 
                        std::vector<cv::Mat>& frames, std::string& cmd_num_frames) {
    m.lock();
    std::cout << CYAN << "-I- Started write thread: " << info_to_str(input_vstream.get_info()) << std::endl << RESET;
    m.unlock();

    hailo_status status = HAILO_SUCCESS;
    
    auto input_shape = input_vstream.get_info().shape;
    int height = input_shape.height;
    int width = input_shape.width;

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

    if (!cmd_num_frames.empty() && input_path.find(".avi") == std::string::npos && input_path.find(".mp4") == std::string::npos){
        capture >> org_frame;
        cv::resize(org_frame, org_frame, cv::Size(width, height), 1);
        status = use_single_frame(input_vstream, write_time_vec, frames, std::ref(org_frame), std::stoi(cmd_num_frames));
        if (HAILO_SUCCESS != status)
            return status;
        capture.release();
    }
    else {
        write_time_vec = std::chrono::high_resolution_clock::now();
        for(;;) {
            capture >> org_frame;
            if(org_frame.empty()) {
                break;
                }
            
            cv::resize(org_frame, org_frame, cv::Size(width, height), 1);
            m.lock();
            frames.push_back(org_frame);
            m.unlock();

            input_vstream.write(MemoryView(frames[frames.size() - 1].data, input_vstream.get_frame_size())); // Writing height * width, 3 channels of uint8
            if (HAILO_SUCCESS != status)
                return status;
            
            org_frame.release();
        }

        capture.release();
    }
    return HAILO_SUCCESS;
}

template <typename T>
hailo_status create_feature(hailo_vstream_info_t vstream_info, size_t output_frame_size, std::shared_ptr<FeatureData<T>> &feature) {
    feature = std::make_shared<FeatureData<T>>(static_cast<uint32_t>(output_frame_size), vstream_info.quant_info.qp_zp,
        vstream_info.quant_info.qp_scale, vstream_info.shape.width, vstream_info);

    return HAILO_SUCCESS;
}

template <typename T>
hailo_status run_inference(std::vector<InputVStream>& input_vstream, std::vector<OutputVStream>& output_vstreams, std::string input_path,
                    std::chrono::time_point<std::chrono::system_clock>& write_time_vec,
                    std::chrono::duration<double>& inference_time, std::chrono::time_point<std::chrono::system_clock>& postprocess_time, 
                    size_t frame_count, double org_height, double org_width, std::string cmd_img_num) {

    hailo_status status = HAILO_UNINITIALIZED;

    std::string model_type = "";
    
    auto output_vstreams_size = output_vstreams.size();

    bool nms_on_hailo = false;
    std::string output_name = (std::string)output_vstreams[0].get_info().name;
    if (output_vstreams_size == 1 && (output_name.find("nms") != std::string::npos)) {
        nms_on_hailo = true;
        model_type = output_name.substr(0, output_name.find('/'));
    }

    std::vector<std::shared_ptr<FeatureData<T>>> features;
    features.reserve(output_vstreams_size);
    for (size_t i = 0; i < output_vstreams_size; i++) {
        std::shared_ptr<FeatureData<T>> feature(nullptr);
        auto status = create_feature(output_vstreams[i].get_info(), output_vstreams[i].get_frame_size(), feature);
        if (HAILO_SUCCESS != status) {
            std::cerr << "Failed creating feature with status = " << status << std::endl;
            return status;
        }

        features.emplace_back(feature);
    }

    std::vector<cv::Mat> frames;

    // Create the write thread
    auto input_thread(std::async(write_all, std::ref(input_vstream[0]), input_path, std::ref(write_time_vec), std::ref(frames), std::ref(cmd_img_num)));

    // Create read threads
    std::vector<std::future<hailo_status>> output_threads;
    output_threads.reserve(output_vstreams_size);
    for (size_t i = 0; i < output_vstreams_size; i++) {
        output_threads.emplace_back(std::async(read_all<T>, std::ref(output_vstreams[i]), features[i], frame_count)); 
    }

    // Create the postprocessing thread
    auto pp_thread(std::async(post_processing_all<T>, std::ref(features), frame_count, std::ref(postprocess_time), std::ref(frames), org_height, org_width, nms_on_hailo, model_type));

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

    inference_time = postprocess_time - write_time_vec;

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

int main(int argc, char** argv) {

    hailo_status status = HAILO_UNINITIALIZED;

    std::chrono::duration<double> total_time;
    std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::high_resolution_clock::now();

    std::string yolov_hef      = getCmdOption(argc, argv, "-hef=");
    std::string input_path      = getCmdOption(argc, argv, "-input=");
    std::string image_num      = getCmdOption(argc, argv, "-num=");

    std::chrono::time_point<std::chrono::system_clock> write_time_vec;
    std::chrono::time_point<std::chrono::system_clock> postprocess_end_time;
    std::chrono::duration<double> inference_time;

    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) {
        std::cerr << "Failed create vdevice, status = " << vdevice_exp.status() << std::endl;
        return vdevice_exp.status();
    }
    auto vdevice = vdevice_exp.release();

    auto network_group_exp = configure_network_group(*vdevice, yolov_hef);
    if (!network_group_exp) {
        std::cerr << "Failed to configure network group " << yolov_hef << std::endl;
        return network_group_exp.status();
    }
    auto network_group = network_group_exp.release();

    auto vstreams_exp = VStreamsBuilder::create_vstreams(*network_group, QUANTIZED, FORMAT_TYPE);
    if (!vstreams_exp) {
        std::cerr << "Failed creating vstreams " << vstreams_exp.status() << std::endl;
        return vstreams_exp.status();
    }
    auto vstreams = vstreams_exp.release();

    print_net_banner(vstreams);

    cv::VideoCapture capture;
    size_t frame_count;
    if (input_path.empty()) {
        capture.open(0, cv::CAP_ANY);
        if (!capture.isOpened()) {
            throw "Error in camera input";
        }
        frame_count = -1;
    }
    else{
        capture.open(input_path, cv::CAP_ANY);
        if (!capture.isOpened()){
            throw "Error when reading video";
        }
        frame_count = (size_t)capture.get(cv::CAP_PROP_FRAME_COUNT);
        if (!image_num.empty() && input_path.find(".avi") == std::string::npos && input_path.find(".mp4") == std::string::npos){
            frame_count = std::stoi(image_num);
        }
    }

    double org_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    double org_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);

    capture.release();

    status = run_inference<uint8_t>(std::ref(vstreams.first), 
                        std::ref(vstreams.second), 
                        input_path, 
                        write_time_vec, inference_time, postprocess_end_time, 
                        frame_count, org_height, org_width, image_num);      

    if (HAILO_SUCCESS != status) {
        std::cerr << "Failed running inference with status = " << status << std::endl;
        return status;
    }

    print_inference_statistics(inference_time, yolov_hef, frame_count);
    std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::high_resolution_clock::now();
    total_time = t_end - t_start;

    std::cout << BOLDBLUE << "\n-I- Application run finished successfully" << RESET << std::endl;
    std::cout << BOLDBLUE << "-I- Total application run time: " << (double)total_time.count() << " sec" << RESET << std::endl;
    return HAILO_SUCCESS;
}
