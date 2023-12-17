#include "hailo/hailort.hpp"
#include "common.h"

#include <iostream>
#include <chrono>
#include <mutex>
#include <future>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>


constexpr bool QUANTIZED = true;
constexpr hailo_format_type_t FORMAT_TYPE_INPUT = HAILO_FORMAT_TYPE_AUTO;
constexpr hailo_format_type_t FORMAT_TYPE_OUTPUT = HAILO_FORMAT_TYPE_AUTO;
std::mutex m;
size_t CAMERA_INPUT_IMAGE_NUM = 300;

using namespace hailort;


void print_inference_statistics(std::chrono::duration<double> inference_time,
                                std::string hef_file, size_t frame_count){
    std::cout << BOLDGREEN << "\n-I-----------------------------------------------" << std::endl;
    std::cout << "-I- " << hef_file.substr(0, hef_file.find(".")) << std::endl;
    std::cout << "-I-----------------------------------------------" << std::endl;
    std::cout << "\n-I-----------------------------------------------" << std::endl;
    std::cout << "-I- Inference                                    " << std::endl;
    std::cout << "-I-----------------------------------------------" << std::endl;
    std::cout << "-I- Total time:   " << inference_time.count() << " sec" << std::endl;
    std::cout << "-I- Average FPS:  " << frame_count / (inference_time.count()) << std::endl;
    std::cout << "-I- Latency:      " << 1.0 / (frame_count / (inference_time.count()) / 1000) << " ms" << std::endl;
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


template <typename T>
hailo_status read_all(OutputVStream& output_vstream, std::vector<cv::Mat>& frames, size_t frame_count, 
                    std::chrono::time_point<std::chrono::system_clock>& read_time_vec) { 

    m.lock();
    std::cout << GREEN << "-I- Started read thread: " << info_to_str(output_vstream.get_info()) << std::endl << RESET;
    m.unlock(); 

    std::vector<T> buffer(output_vstream.get_frame_size());

    for (size_t i = 0; i < frame_count; i++) {
        hailo_status status = output_vstream.read(MemoryView(buffer.data(), buffer.size()));
        cv::Mat imageMat(output_vstream.get_info().shape.height, output_vstream.get_info().shape.width, CV_8U, buffer.data());
        //// Display
        // cv::imshow("Display window", frames[0]);
        // cv::waitKey(0);

        //// Save inferred image
        cv::imwrite("output_image.jpg", imageMat);
        if (HAILO_SUCCESS != status) {
            std::cerr << "Failed reading with status = " <<  status << std::endl;
            return status;
        }
    }

    read_time_vec = std::chrono::high_resolution_clock::now();
    return HAILO_SUCCESS;
}

cv::Mat stereonet_preprocess(cv::Mat image, int target_height, int target_width) {
    int height = image.rows;
    int width = image.cols;

    // Calculate the amount of padding required
    int pad_height = std::max(target_height - height, 0);
    int pad_width = std::max(target_width - width, 0);

    // Pad the matrix symmetrically on all sides
    cv::Mat padded_image;
    cv::copyMakeBorder(image, padded_image, 0, pad_height, 0, pad_width, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    // Crop or pad the matrix to the target shape
    cv::Rect region_of_interest(0, 0, target_width, target_height);
    cv::Mat cropped_image = padded_image(region_of_interest).clone();

    return cropped_image;
}


hailo_status write_all(InputVStream& input_vstream, std::vector<cv::Mat>& frames, 
                        std::string input_path, std::chrono::time_point<std::chrono::system_clock>& write_time_vec) {
    m.lock();
    std::cout << CYAN << "-I- Started write thread: " << info_to_str(input_vstream.get_info()) << std::endl << RESET;
    m.unlock();

    hailo_status status = HAILO_SUCCESS;
    
    auto input_shape = input_vstream.get_info().shape;
    int height = input_shape.height;
    int width = input_shape.width;

    cv::VideoCapture capture(input_path);
    if(!capture.isOpened())
        throw "Unable to read video file";
    
    int i = 0;
    cv::Mat org_frame;
    cv::Mat frame;

    write_time_vec = std::chrono::high_resolution_clock::now();
    for(;;) {
        capture >> org_frame;
        if(org_frame.empty()) {
            break;
            }
        cv::cvtColor(org_frame, org_frame, cv::COLOR_BGR2RGB);

        frames[i] = stereonet_preprocess(org_frame, height, width);

        input_vstream.write(MemoryView(frames[i].data, input_vstream.get_frame_size())); // Writing height * width, 3 channels of uint8
        if (HAILO_SUCCESS != status)
            return status;
        i++;
    }

    capture.release();
    return HAILO_SUCCESS;
}


template <typename T>
hailo_status run_inference(std::vector<InputVStream>& input_vstreams, std::vector<OutputVStream>& output_vstreams, 
                    std::string right_input, std::string left_input,
                    std::vector<std::chrono::time_point<std::chrono::system_clock>>& write_time_vec,
                    std::vector<std::chrono::time_point<std::chrono::system_clock>>& read_time_vec,
                    std::chrono::duration<double>& inference_time, size_t frame_count) {

    hailo_status input_status = HAILO_UNINITIALIZED;
    hailo_status status = HAILO_UNINITIALIZED;

    std::vector<cv::Mat> frames(frame_count);
    
    auto input_vstreams_size = input_vstreams.size();

    std::map<std::string, std::string> input_name_to_images_path;

    for (auto& inp_vstm : input_vstreams) {
        std::string input_name = inp_vstm.get_info().name;
        if (input_name.find("input_layer1") != std::string::npos)
            input_name_to_images_path[input_name] =  left_input;
        else
            input_name_to_images_path[input_name] = right_input;
    }
    

    // Create write threads
    std::vector<std::future<hailo_status>> input_threads;
    input_threads.reserve(input_vstreams_size);
    for (size_t i = 0; i < input_vstreams_size; i++) {
        input_threads.emplace_back(std::async(write_all, std::ref(input_vstreams[i]), 
                                            std::ref(frames), input_name_to_images_path.at(input_vstreams[i].get_info().name), 
                                            std::ref(write_time_vec[i]))); 
    }

    auto output_thread(std::async(read_all<T>, std::ref(output_vstreams[0]), std::ref(frames), frame_count, std::ref(read_time_vec[0])));

    for (size_t i = 0; i < input_threads.size(); i++) {
        input_status = input_threads[i].get();
    }
   
    status = output_thread.get();

    if (HAILO_SUCCESS != input_status) {
        std::cerr << "Write thread failed with status " << input_status << std::endl;
        return input_status; 
    }
    if (HAILO_SUCCESS != status) {
        std::cerr << "Read failed with status " << status << std::endl;
        return status;
    }

    inference_time = read_time_vec[0] - write_time_vec[0];
    for (size_t i = 1; i < input_vstreams.size(); i++){
        if (inference_time.count() < (double)(read_time_vec[0] - write_time_vec[i]).count())
            inference_time = read_time_vec[0] - write_time_vec[i];
    }

    std::cout << BOLDBLUE << "\n-I- Inference finished successfully" << RESET << std::endl;

    status = HAILO_SUCCESS;
    return status;
}

size_t get_frame_count(std::string input_path){
    cv::VideoCapture capture(input_path);
    if (!capture.isOpened()){
        throw "Error when reading video";
    }
    double frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);
    capture.release();
    return (size_t)frame_count;
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

    std::string stereonet_hef   = getCmdOption(argc, argv, "-hef=");
    std::string right_input      = getCmdOption(argc, argv, "-right=");
    std::string left_input      = getCmdOption(argc, argv, "-left=");

    std::chrono::duration<double> inference_time;

    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) {
        std::cerr << "Failed create vdevice, status = " << vdevice_exp.status() << std::endl;
        return vdevice_exp.status();
    }
    auto vdevice = vdevice_exp.release();

    auto network_group = configure_network_group(*vdevice, stereonet_hef);
    if (!network_group) {
        std::cerr << "Failed to configure network group " << stereonet_hef << std::endl;
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

    std::vector<std::chrono::time_point<std::chrono::system_clock>> write_time_vec(vstreams.first.size());
    std::vector<std::chrono::time_point<std::chrono::system_clock>> read_time_vec(vstreams.second.size());

    print_net_banner(vstreams);

    size_t frame_count;

    if (!right_input.empty() && !left_input.empty()) {
        frame_count = get_frame_count(right_input);

        if (frame_count != get_frame_count(left_input)){
            std::cerr << "Number of images of left and right inputs must be equal" << std::endl;
        }
    }
    else if (right_input.empty() && left_input.empty()){
        frame_count = CAMERA_INPUT_IMAGE_NUM;
    }
    else{
        std::cerr << "Please define inputs for both left and right" << std::endl;
    }

    status = run_inference<uint8_t>(std::ref(vstreams.first), 
                                    std::ref(vstreams.second), 
                                    right_input, left_input,
                                    write_time_vec, read_time_vec, 
                                    inference_time, frame_count);

    if (HAILO_SUCCESS != status) {
        std::cerr << "Failed running inference with status = " << status << std::endl;
        return status;
    }

    print_inference_statistics(inference_time, stereonet_hef, frame_count);

    std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::high_resolution_clock::now();
    total_time = t_end - t_start;

    std::cout << BOLDBLUE << "\n-I- Application run finished successfully" << RESET << std::endl;
    std::cout << BOLDBLUE << "-I- Total application run time: " << (double)total_time.count() << " sec" << RESET << std::endl;
    return HAILO_SUCCESS;
}
