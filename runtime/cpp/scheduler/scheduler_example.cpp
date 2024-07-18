#include "hailo/hailort.hpp"
#include <iostream>
#include <chrono>
#include <mutex>
#include <thread>

#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

constexpr bool QUANTIZED = true;
constexpr hailo_format_type_t FORMAT_TYPE = HAILO_FORMAT_TYPE_AUTO;
constexpr uint32_t DEVICE_COUNT = 1;
std::mutex m;

using namespace hailort;
using ThreadsVector = std::vector<std::unique_ptr<std::thread>>;
using StatusVector = std::vector<std::shared_ptr<hailo_status>>;
using hailort::Device;
using hailort::Hef;
using hailort::Expected;
using hailort::make_unexpected;
using hailort::ConfiguredNetworkGroup;
using hailort::VStreamsBuilder;
using hailort::InputVStream;
using hailort::OutputVStream;
using hailort::MemoryView;


void print_inference_statistics(std::vector<std::string> num_of_frames, std::vector<std::chrono::duration<double>> inference_time, 
                                std::vector<std::string> hef_files){
    for (std::size_t i = 0; i < hef_files.size(); i++) {
        std::cout << BOLDGREEN << "\n-I-----------------------------------------------" << std::endl;
        std::cout << "-I- " << hef_files[i].substr(0, hef_files[i].find(".hef")) << std::endl;
        std::cout << "-I-----------------------------------------------" << std::endl;
        std::cout << "-I- Total time:   " << inference_time[i].count() << " sec" << std::endl;
        std::cout << "-I- Average FPS:  " << (double)stoi(num_of_frames[i]) / (inference_time[i].count()) << std::endl;
        std::cout << "-I- Latency:      " << 1.0 / ((double)stoi(num_of_frames[i]) / (inference_time[i].count()) / 1000) << " ms" << std::endl;
        std::cout << "-I-----------------------------------------------" << std::endl << RESET;
    }
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


template<typename T=uint8_t>
void read_all(OutputVStream &output_vstream, std::shared_ptr<hailo_status> status_out, 
                std::string frame_count, std::vector<std::chrono::time_point<std::chrono::system_clock>>& read_time_vec, 
                std::vector<std::chrono::time_point<std::chrono::system_clock>>& write_time_vec,
                std::vector<std::chrono::duration<double>>& inference_time_vec, int hef_id) {
    m.lock();
    std::cout << GREEN << "-I- Started read thread: " << info_to_str(output_vstream.get_info()) << std::endl << RESET;
    m.unlock();
    std::vector<T> buff(output_vstream.get_frame_size());
    std::size_t num_of_frames = (size_t)stoi(frame_count);
    for (size_t i = 0; i < num_of_frames; i++) {
        auto status = output_vstream.read(MemoryView(buff.data(), buff.size()));
        if (HAILO_SUCCESS != status) {
            *status_out = status;
            return;
        }
    }
    read_time_vec[hef_id] = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> curr_inference_time = read_time_vec[hef_id] - write_time_vec[hef_id];
    if (inference_time_vec[hef_id].count() == 0 || inference_time_vec[hef_id].count() < curr_inference_time.count()){
        inference_time_vec[hef_id] = read_time_vec[hef_id] - write_time_vec[hef_id];
    }
    std::cout << YELLOW << "-I- " << output_vstream.get_info().name << " Recived " << frame_count << " Images" << std::endl << RESET;
    *status_out = HAILO_SUCCESS;
    return;
}


template<typename T=uint8_t>
void write_all(InputVStream &input_vstream, std::shared_ptr<hailo_status> status_out, 
                std::string frame_count, std::vector<std::chrono::time_point<std::chrono::system_clock>>& write_time_vec, int hef_id) {
    m.lock();
    std::cout << CYAN << "-I- Started write thread: " << info_to_str(input_vstream.get_info()) << std::endl << RESET;
    m.unlock();
    std::vector<T> buff(input_vstream.get_frame_size());
    write_time_vec[hef_id] = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < (size_t)stoi(frame_count); i++) {
        auto status = input_vstream.write(MemoryView(buff.data(), buff.size()));
        if (HAILO_SUCCESS != status) {
            *status_out = status;
            return;
        }
    }
    *status_out = HAILO_SUCCESS;
    return;
}


void create_read_threads(std::vector<OutputVStream> &vstreams, StatusVector &read_results, 
                        ThreadsVector &threads_vector, std::string frame_count, 
                        std::vector<std::chrono::time_point<std::chrono::system_clock>>& read_time_vec,  
                        std::vector<std::chrono::time_point<std::chrono::system_clock>>& write_time_vec, 
                        std::vector<std::chrono::duration<double>>& inference_time_vec, int hef_id) {
    for (auto &vstream : vstreams) {
        read_results.push_back(std::make_shared<hailo_status>(HAILO_UNINITIALIZED));
        threads_vector.emplace_back(std::make_unique<std::thread>(read_all, std::ref(vstream), read_results.back(), 
                                    frame_count, std::ref(read_time_vec), std::ref(write_time_vec), std::ref(inference_time_vec), hef_id));
    }
}

void create_write_threads(std::vector<InputVStream> &vstreams, StatusVector &write_results, ThreadsVector &threads_vector, 
                        std::string frame_count, std::vector<std::chrono::time_point<std::chrono::system_clock>>& write_time_vec, int hef_id) {
    
    for (auto &vstream : vstreams) {
        write_results.push_back(std::make_shared<hailo_status>(HAILO_UNINITIALIZED));
        threads_vector.emplace_back(std::make_unique<std::thread>(write_all, std::ref(vstream), write_results.back(), 
                                    frame_count, std::ref(write_time_vec), hef_id));
    }
}


void print_net_banner(std::pair<std::vector<hailort::InputVStream>, std::vector<hailort::OutputVStream>> &vstreams, std::string frames_count) {
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
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
    std::cout << BOLDMAGENTA << "-I-  Number Of Frames                            " << std::endl << RESET;
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
    std::cout << MAGENTA << "-I-  " << frames_count << std::endl << RESET;
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


Expected<std::vector<std::shared_ptr<ConfiguredNetworkGroup>>> configure_hefs(VDevice &vdevice, std::vector<std::string> &hef_paths)
{
    std::vector<std::shared_ptr<ConfiguredNetworkGroup>> results;

    for (const auto &path : hef_paths) {
        auto hef_exp = Hef::create(path);
        if (!hef_exp) {
            return make_unexpected(hef_exp.status());
        }
        auto hef = hef_exp.release();

        auto added_network_groups = vdevice.configure(hef);
        if (!added_network_groups) {
            return make_unexpected(added_network_groups.status());
        }
        results.insert(results.end(), added_network_groups->begin(),
            added_network_groups->end());
    }
    return results;
}


Expected<std::unique_ptr<VDevice>> create_vdevice()
{
    hailo_vdevice_params_t params;
    auto status = hailo_init_vdevice_params(&params);
    if (HAILO_SUCCESS != status) {
        std::cerr << "Failed init vdevice_params, status = " << status << std::endl;
        return make_unexpected(status);
    }
    params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
    params.device_count = DEVICE_COUNT;

    params.multi_process_service = true;
    params.group_id = "SHARED";

    return VDevice::create(params);
}

void get_all_args(std::string s, std::vector<std::string> &v){
	std::string temp = "";
	for (size_t i = 0; i < s.length(); i++){
		if(s[i] == ','){
			v.push_back(temp);
			temp = "";
		}
		else {
			temp.push_back(s[i]);
		}
	}
	v.push_back(temp);
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
    if (cmd.empty() && option=="-num=")
        return "100";
    return cmd;
}

int main(int argc, char** argv) {

    std::chrono::duration<double> total_time;
    std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::high_resolution_clock::now();

    std::string all_hefs      = getCmdOption(argc, argv, "-hefs=");
    std::string num_of_frames = getCmdOption(argc, argv, "-num=");

    std::vector<std::string> hef_files;
    std::vector<std::string> frames_count;
    get_all_args(all_hefs, hef_files);
    if (num_of_frames.compare("100") == 0){
        for (size_t i = 0; i < hef_files.size(); i++){
            frames_count.push_back("100");
        }
    }
    else {
        get_all_args(num_of_frames, frames_count);
    }

    if (hef_files.size() == 0 || frames_count.size()==0 || hef_files.size() != frames_count.size()){
        std::cerr << BOLDRED << "Wrong input. Please specify the same number of HEFs names and number of frames." << std::endl << RESET;
        exit(1);
    }

    std::vector<std::chrono::time_point<std::chrono::system_clock>> read_time_vec(hef_files.size());
    std::vector<std::chrono::time_point<std::chrono::system_clock>> write_time_vec(hef_files.size());
    std::vector<std::chrono::duration<double>> inference_time_vec(hef_files.size());

    auto scan_res = hailort::Device::scan();
    if (!scan_res) {
        std::cerr << "Failed to scan, status = " << scan_res.status() << std::endl;
        return scan_res.status();
    }
    std::cout << "Found " << scan_res.value().size() << " devices" << std::endl;

    auto vdevice_exp = create_vdevice();
    if (!vdevice_exp) {
        std::cerr << "Failed create vdevice, status = " << vdevice_exp.status() << std::endl;
        return vdevice_exp.status();
    }
    auto vdevice = vdevice_exp.release();

    auto configured_network_groups_exp = configure_hefs(*vdevice, hef_files);
    if (!configured_network_groups_exp) {
        std::cerr << "Failed to configure HEFs, status = " << configured_network_groups_exp.status() << std::endl;
        return configured_network_groups_exp.status();
    }
    auto configured_network_groups = configured_network_groups_exp.release();

    auto vstreams_per_network_group_exp = build_vstreams(configured_network_groups);
    if (!vstreams_per_network_group_exp) {
        std::cerr << "Failed to create vstreams, status = " << vstreams_per_network_group_exp.status() << std::endl;
        return vstreams_per_network_group_exp.status();
    }

    auto vstreams_per_network_group = vstreams_per_network_group_exp.release();

    int index = 0;

    for (auto &vstreams_pair : vstreams_per_network_group) {
        print_net_banner(vstreams_pair, frames_count[index]);
        index++;
    }

    index = 0;

    ThreadsVector threads;
    StatusVector results;

    for (auto &vstreams_pair : vstreams_per_network_group) {
        // Create send/recv threads
        create_write_threads(vstreams_pair.first, results, threads, frames_count[index], write_time_vec, index);
        create_read_threads(vstreams_pair.second, results, threads, frames_count[index], read_time_vec, write_time_vec, inference_time_vec, index);
        index++;
    }

    // Join threads and validate results
    for (auto &thread : threads) {
        if (thread->joinable()) {
            thread->join();
        }
    }
    print_inference_statistics(frames_count, inference_time_vec, hef_files);

    std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::high_resolution_clock::now();
    total_time = t_end - t_start;

    for (auto &status : results) {
        if (HAILO_SUCCESS != *status) {
            std::cerr << "\nInference failed, status = "  << *status << std::endl;
            return *status;
        }
    }

    std::cout << BOLDBLUE << "\n-I- Inference finished successfully" << RESET << std::endl;
    std::cout << BOLDBLUE << "-I- Total inference run time: " << (double)total_time.count() << " sec" << RESET << std::endl;
    return HAILO_SUCCESS;
}
