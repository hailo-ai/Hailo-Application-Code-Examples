#include "hailo/hailort.hpp"
#include "common.h"
#include "common/yolo_postprocess.hpp"
#include "common/yolo_output.hpp"
#include "common/yolo_hailortpp.hpp"
#include "common/labels/coco_ninety.hpp"

#include <iostream>
#include <chrono>
#include <mutex>
#include <future>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>


constexpr bool QUANTIZED = true;
constexpr hailo_format_type_t FORMAT_TYPE_INPUT = HAILO_FORMAT_TYPE_AUTO;
constexpr hailo_format_type_t FORMAT_TYPE_OUTPUT = HAILO_FORMAT_TYPE_AUTO;
std::mutex m;
std::string model_arch;
bool is_camera = false;

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

void postprocess_nms_on_hailo(HailoROIPtr& roi, bool nms_on_hailo, std::string output_name) {
    if (nms_on_hailo)
        filter_nms(roi, output_name);
    else {
        std::string config = model_arch + ".json";
        YoloParams *init_params = init(config, model_arch);
        filter(roi, init_params);
    }
}

template <typename T>
hailo_status post_processing_all(std::vector<std::shared_ptr<FeatureData<T>>> &features, size_t  frame_count, 
                                std::chrono::duration<double>& postprocess_time, std::vector<cv::Mat>& frames, 
                                double org_height, double org_width, bool nms_on_hailo, std::string output_name, 
                                bool save_output, bool display_images)
{
    auto status = HAILO_SUCCESS;   
    std::sort(features.begin(), features.end(), &FeatureData<T>::sort_tensors_by_size);

    cv::VideoWriter video;
    if (save_output) {
        video = cv::VideoWriter("./processed_video.mp4", cv::VideoWriter::fourcc('m','p','4','v'),30, cv::Size((int)org_width, (int)org_height));
    }
    std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::high_resolution_clock::now();

    m.lock();
    std::cout << YELLOW << "\n-I- Starting postprocessing\n" << std::endl << RESET;
    m.unlock();

    for (size_t i = 0; i < frame_count; i++){
        HailoROIPtr roi = std::make_shared<HailoROI>(HailoROI(HailoBBox(0.0f, 0.0f, 1.0f, 1.0f)));
        for (uint j = 0; j < features.size(); j++) {
            roi->add_tensor(std::make_shared<HailoTensor>(reinterpret_cast<T *>(features[j]->m_buffers.get_read_buffer().data()), features[j]->m_vstream_info));
        }
        postprocess_nms_on_hailo(std::ref(roi), nms_on_hailo, output_name);

        for (auto &feature : features) {
            feature->m_buffers.release_read_buffer();
        }
        std::vector<HailoDetectionPtr> detections = hailo_common::get_hailo_detections(roi);

        cv::resize(frames[0], frames[0], cv::Size((int)org_width, (int)org_height), 1);
        for (auto &detection : detections) {
            if (detection->get_confidence() == 0) {
                continue;
            }
            auto box = detection->get_bbox();
            cv::rectangle(frames[0], cv::Point2f(float(box.xmin() * float(org_width)), float(box.ymin() * float(org_height))), 
                        cv::Point2f(float(box.xmax() * float(org_width)), float(box.ymax() * float(org_height))), 
                        cv::Scalar(0, 0, 255), 1);

            std::cout << "Detection: " << get_coco_name_from_int(detection->get_class_id()) << ", Confidence: " << std::fixed << std::setprecision(2) << detection->get_confidence() * 100.0 << "%" << std::endl;
        }
        if (display_images) {
            cv::imshow("Display window", frames[i]);
            cv::waitKey(0);
        }
        if (save_output){
            video.write(frames[0]);
        }

        frames[0].release();
        
        m.lock();
        frames.erase(frames.begin());
        m.unlock();
        if (is_camera && frame_count == 1){
            i--;
        }
    }
    std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::high_resolution_clock::now();
    postprocess_time = t_end - t_start;
    if (save_output){
        video.release();
    }

    return status;
}


template <typename T>
hailo_status read_all(OutputVStream& output_vstream, std::shared_ptr<FeatureData<T>> feature, size_t frame_count, 
                    std::chrono::time_point<std::chrono::system_clock>& read_time_vec) { 

    m.lock();
    std::cout << GREEN << "-I- Started read thread: " << info_to_str(output_vstream.get_info()) << std::endl << RESET;
    m.unlock();

    for (size_t i = 0; i < frame_count; i++) {
        std::vector<T>& buffer = feature->m_buffers.get_write_buffer();
        hailo_status status = output_vstream.read(MemoryView(buffer.data(), buffer.size()));
        feature->m_buffers.release_write_buffer();
        if (HAILO_SUCCESS != status) {
            std::cerr << "Failed reading with status = " <<  status << std::endl;
            return status;
        }

        if (is_camera && frame_count == 1){
            i--;
        }
    }

    read_time_vec = std::chrono::high_resolution_clock::now();
    return HAILO_SUCCESS;
}


hailo_status use_single_frame(InputVStream& input_vstream, std::chrono::time_point<std::chrono::system_clock>& write_time_vec,
                                std::vector<cv::Mat>& frames, cv::Mat& image, size_t frame_count){
    hailo_status status = HAILO_SUCCESS;
    write_time_vec = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < frame_count; i++) {
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
                        std::chrono::time_point<std::chrono::system_clock>& write_time_vec, std::vector<cv::Mat>& frames, 
                        size_t frame_count, bool single_image) {
    m.lock();
    std::cout << CYAN << "-I- Started write thread: " << info_to_str(input_vstream.get_info()) << std::endl << RESET;
    m.unlock();

    hailo_status status = HAILO_SUCCESS;
    
    auto input_shape = input_vstream.get_info().shape;
    int height = input_shape.height;
    int width = input_shape.width;
    cv::Mat org_frame;
    cv::VideoCapture capture;

    if (single_image)
    {
        capture.open(input_path, cv::CAP_ANY);
        if(!capture.isOpened()){
            throw "Unable to read input file";
        }
        capture >> org_frame;
        cv::resize(org_frame, org_frame, cv::Size(width, height), 1);
        status = use_single_frame(input_vstream, write_time_vec, frames, std::ref(org_frame), frame_count);
        if (HAILO_SUCCESS != status)
            return status;
        capture.release();
    }
    else {
        is_camera ? capture.open(0, cv::CAP_ANY) : capture.open(input_path, cv::CAP_ANY);
        if (!capture.isOpened()){
            throw "Unable to read video or camera input";
        }

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
                    std::vector<std::chrono::time_point<std::chrono::system_clock>>& read_time_vec,
                    std::chrono::duration<double>& inference_time, std::chrono::duration<double>& postprocess_time, 
                    size_t frame_count, double org_height, double org_width, bool save_output, bool display_images, bool single_image) {

    hailo_status status = HAILO_UNINITIALIZED;
    
    auto output_vstreams_size = output_vstreams.size();

    bool nms_on_hailo = false;
    std::string output_name = "";

    if (output_vstreams_size == 1 && ((std::string)output_vstreams[0].get_info().name).find("nms") != std::string::npos) {
        nms_on_hailo = true;
        output_name = (std::string)output_vstreams[0].get_info().name;
    }
    std::vector<std::shared_ptr<FeatureData<T>>> features;
    features.reserve(output_vstreams_size);
    for (size_t i = 0; i < output_vstreams_size; i++) {
        std::shared_ptr<FeatureData<T>> feature(nullptr);
        auto status = create_feature<T>(output_vstreams[i].get_info(), output_vstreams[i].get_frame_size(), feature);
        if (HAILO_SUCCESS != status) {
            std::cerr << "Failed creating feature with status = " << status << std::endl;
            return status;
        }

        features.emplace_back(feature);
    }

    std::vector<cv::Mat> frames;

    auto input_thread(std::async(write_all, std::ref(input_vstream[0]), input_path, std::ref(write_time_vec), std::ref(frames), frame_count, single_image));

    // Create read threads
    std::vector<std::future<hailo_status>> output_threads;
    output_threads.reserve(output_vstreams_size);
    for (size_t i = 0; i < output_vstreams_size; i++) {
        output_threads.emplace_back(std::async(read_all<T>, std::ref(output_vstreams[i]), features[i], frame_count, std::ref(read_time_vec[i]))); 
    }

    auto pp_thread(std::async(post_processing_all<T>, std::ref(features), frame_count, std::ref(postprocess_time), 
                                                        std::ref(frames), org_height, org_width, nms_on_hailo, output_name, 
                                                        save_output, display_images));

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

size_t get_frame_rate(cv::VideoCapture& capture, const std::string& input_path, const std::string &image_num, bool &single_image)
{
    if (input_path.find(".avi") == std::string::npos && input_path.find(".mp4") == std::string::npos) { // Running single image or via camera
        if (input_path.empty())
            is_camera = true;
        else
            single_image = true;

        return image_num.empty() ? 1 : std::stoi(image_num);
    }
    else { //Video
        return -1;
    }
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

std::string get_cmd_options(int argc, char *argv[], const std::string &option)
{
    std::string cmd = "";
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (0 == arg.find(option, 0))
        {
            if (-1 == (int)arg.find("=") && ((0 == arg.find("-out", 0)) || (0 == arg.find("-display", 0)))) 
                return "true";
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

    std::string yolo_hef        = get_cmd_options(argc, argv, "-hef=");
    std::string input_path      = get_cmd_options(argc, argv, "-input=");
    model_arch                  = get_cmd_options(argc, argv, "-arch=");
    bool save_output            = get_cmd_options(argc, argv, "-out").compare("true") == 0 ? true : false;
    bool display_images         = get_cmd_options(argc, argv, "-display").compare("true") == 0 ? true : false;
    std::string image_num       = get_cmd_options(argc, argv, "-num=");
    bool single_image           = false;
    
    if (yolo_hef == "" || model_arch == "")
    {
        std::cerr << "Please provide valid HEF file and corresponding architecture to run the application." << std::endl;
        return HAILO_INVALID_ARGUMENT;
    }

    std::chrono::time_point<std::chrono::system_clock> write_time_vec;
    std::chrono::duration<double> inference_time;
    std::chrono::duration<double> postprocess_time;

    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) {
        std::cerr << "Failed create vdevice, status = " << vdevice_exp.status() << std::endl;
        return vdevice_exp.status();
    }
    auto vdevice = vdevice_exp.release();

    auto network_group = configure_network_group(*vdevice, yolo_hef);
    if (!network_group) {
        std::cerr << "Failed to configure network group " << yolo_hef << std::endl;
        return network_group.status();
    }

    auto input_vstreams_params = network_group.value()->make_input_vstream_params(QUANTIZED, FORMAT_TYPE_INPUT, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    if (!input_vstreams_params) {
        std::cerr << "Failed creating input vstreams " << input_vstreams_params.status() << std::endl;
        return input_vstreams_params.status();
    }

    auto output_vstreams_params = network_group.value()->make_output_vstream_params(QUANTIZED, FORMAT_TYPE_OUTPUT, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    if (!output_vstreams_params) {
        std::cerr << "Failed creating vstreams " << output_vstreams_params.status() << std::endl;
        return output_vstreams_params.status();
    }

    auto input_vstreams = VStreamsBuilder::create_input_vstreams(*network_group.value(), *input_vstreams_params);
    if (!input_vstreams) {
        std::cerr << "Failed creating input vstreams " << input_vstreams.status() << std::endl;
        return input_vstreams.status();
    }

    auto output_vstreams = VStreamsBuilder::create_output_vstreams(*network_group.value(), *output_vstreams_params);
    if (!output_vstreams) {
        std::cerr << "Failed creating output vstreams " << output_vstreams.status() << std::endl;
        return output_vstreams.status();
    }

    auto vstreams = std::make_pair(input_vstreams.release(), output_vstreams.release());

    std::vector<std::chrono::time_point<std::chrono::system_clock>> read_time_vec(vstreams.second.size());

    print_net_banner(vstreams);

    cv::VideoCapture capture;
    size_t frame_count = get_frame_rate(capture, input_path, image_num, single_image);
    is_camera ? capture.open(0, cv::CAP_ANY) : capture.open(input_path, cv::CAP_ANY);;
    if (frame_count == -1){
        frame_count = (size_t)capture.get(cv::CAP_PROP_FRAME_COUNT);
    }
    double org_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    double org_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    capture.release();

    status = run_inference<uint8_t>(std::ref(vstreams.first), 
                                    std::ref(vstreams.second), 
                                    input_path, 
                                    write_time_vec, read_time_vec, 
                                    inference_time, postprocess_time, 
                                    frame_count, org_height, org_width, 
                                    save_output, display_images, single_image);


    if (HAILO_SUCCESS != status) {
        std::cerr << "Failed running inference with status = " << status << std::endl;
        return status;
    }

    print_inference_statistics(inference_time, postprocess_time, yolo_hef, static_cast<double>(frame_count));

    std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::high_resolution_clock::now();
    total_time = t_end - t_start;

    std::cout << BOLDBLUE << "\n-I- Application run finished successfully" << RESET << std::endl;
    std::cout << BOLDBLUE << "-I- Total application run time: " << (double)total_time.count() << " sec" << RESET << std::endl;
    return HAILO_SUCCESS;
}
