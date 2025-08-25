#include "toolbox.hpp"
#include "hailo_infer.hpp"
namespace hailo_utils {
hailo_status check_status(const hailo_status &status, const std::string &message) {
    if (HAILO_SUCCESS != status) {
        std::cerr << message << " with status " << status << std::endl;
        return status;
    }
    return HAILO_SUCCESS;
}

hailo_status wait_and_check_threads(
    std::future<hailo_status> &f1, const std::string &name1,
    std::future<hailo_status> &f2, const std::string &name2,
    std::future<hailo_status> &f3, const std::string &name3)
{
    hailo_status status = f1.get();
    if (HAILO_SUCCESS != status) {
        std::cerr << name1 << " failed with status " << status << std::endl;
        return status;
    }

    status = f2.get();
    if (HAILO_SUCCESS != status) {
        std::cerr << name2 << " failed with status " << status << std::endl;
        return status;
    }

    status = f3.get();
    if (HAILO_SUCCESS != status) {
        std::cerr << name3 << " failed with status " << status << std::endl;
        return status;
    }

    return HAILO_SUCCESS;
}

bool is_image_file(const std::string& path) {
    static const std::vector<std::string> image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"};
    std::string extension = fs::path(path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    return std::find(image_extensions.begin(), image_extensions.end(), extension) != image_extensions.end();
}

bool is_video_file(const std::string& path) {
    static const std::vector<std::string> video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"};
    std::string extension = fs::path(path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    return std::find(video_extensions.begin(), video_extensions.end(), extension) != video_extensions.end();
}

bool is_directory_of_images(const std::string& path, int &entry_count, size_t batch_size) {
    entry_count = 0;
    if (fs::exists(path) && fs::is_directory(path)) {
        bool has_images = false;
        for (const auto& entry : fs::directory_iterator(path)) {
            if (fs::is_regular_file(entry)) {
                entry_count++;
                if (!is_image_file(entry.path().string())) {
                    // Found a non-image file
                    return false;
                }
                has_images = true; 
            }
        }
        if (entry_count % batch_size != 0) {
            throw std::invalid_argument("Directory contains " + std::to_string(entry_count) + " images, which is not divisible by batch size " + std::to_string(batch_size));
        }
        return has_images; 
    }
    return false;
}

bool is_image(const std::string& path) {
    return fs::exists(path) && fs::is_regular_file(path) && is_image_file(path);
}

bool is_video(const std::string& path) {
    return fs::exists(path) && fs::is_regular_file(path) && is_video_file(path);
}

std::string get_hef_name(const std::string &path)
{
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
}


std::string getCmdOption(int argc, char *argv[], const std::string &option)
{
    std::string cmd;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (0 == arg.find(option, 0)) {
            std::size_t found = arg.find("=", 0) + 1;
            cmd = arg.substr(found);
            return cmd;
        }
    }
    return cmd;
}

bool has_flag(int argc, char *argv[], const std::string &flag) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == flag) {
            return true;
        }
    }
    return false;
}
std::string getCmdOptionWithShortFlag(int argc, char *argv[], const std::string &longOption, const std::string &shortOption) {
    std::string cmd;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == longOption || arg == shortOption) {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                cmd = argv[i + 1];
                return cmd;
            }
        }
    }
    return cmd;
}

CommandLineArgs parse_command_line_arguments(int argc, char** argv) {
    return {
        getCmdOptionWithShortFlag(argc, argv, "--net", "-n"),
        getCmdOptionWithShortFlag(argc, argv, "--input", "-i"),
        has_flag(argc, argv, "-s"),
        (getCmdOptionWithShortFlag(argc, argv, "--batch_size", "-b").empty() ? "1" : getCmdOptionWithShortFlag(argc, argv, "--batch_size", "-b"))
    };
}

InputType determine_input_type(const std::string& input_path, cv::VideoCapture &capture,
                               double &org_height, double &org_width, size_t &frame_count, size_t batch_size) {

    InputType input_type;
    int directory_entry_count;
    if (is_directory_of_images(input_path, directory_entry_count, batch_size)) {
        input_type.is_directory = true;
        input_type.directory_entry_count = directory_entry_count;
    } else if (is_image(input_path)) {
        input_type.is_image = true;
    } else if (is_video(input_path)) {
        input_type.is_video = true;
        capture = open_video_capture(input_path, std::ref(capture), org_height, org_width, frame_count);
    } else {
        std::cout << "Input is not an image or video, trying to open as camera" << std::endl;
        input_type.is_camera = true;
        capture = open_video_capture(input_path, std::ref(capture), org_height, org_width, frame_count);
    }
    return input_type;
}

void show_progress_helper(size_t current, size_t total)
{
    int progress = static_cast<int>((static_cast<float>(current + 1) / static_cast<float>(total)) * 100);
    int bar_width = 50; 
    int pos = static_cast<int>(bar_width * (current + 1) / total);

    std::cout << "\rProgress: [";
    for (int j = 0; j < bar_width; ++j) {
        if (j < pos) std::cout << "=";
        else if (j == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::setw(3) << progress
              << "% (" << std::setw(3) << (current + 1) << "/" << total << ")" << std::flush;
}

void show_progress(hailo_utils::InputType &input_type, int progress, size_t frame_count) {
    if (input_type.is_video) {
        show_progress_helper(progress, frame_count);
    } else if (input_type.is_directory) {
        show_progress_helper(progress, input_type.directory_entry_count);
    }
}

void print_inference_statistics(std::chrono::duration<double> inference_time,
    const std::string &hef_file,
    double frame_count,
    std::chrono::duration<double> total_time)
{
    std::cout << BOLDGREEN << "\n-I-----------------------------------------------" << std::endl;
    std::cout << "-I- Inference & Postprocess                        " << std::endl;
    std::cout << "-I-----------------------------------------------" << std::endl;
    std::cout << "-I- Average FPS:  " << frame_count / (inference_time.count()) << std::endl;
    std::cout << "-I- Total time:   " << inference_time.count() << " sec" << std::endl;
    std::cout << "-I- Latency:      "
    << 1.0 / (frame_count / (inference_time.count()) / 1000) << " ms" << std::endl;
    std::cout << "-I-----------------------------------------------" << std::endl;

    std::cout << BOLDBLUE << "\n-I- Application finished successfully" << RESET << std::endl;
    std::cout << BOLDBLUE << "-I- Total application run time: " << (double)total_time.count() << " sec" << RESET << std::endl;
}

void print_net_banner(const std::string &detection_model_name,
    const std::vector<hailort::InferModel::InferStream> &inputs,
    const std::vector<hailort::InferModel::InferStream> &outputs)
{
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
    std::cout << BOLDMAGENTA << "-I-  Network Name                               " << std::endl << RESET;
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
    std::cout << BOLDMAGENTA << "-I   " << detection_model_name << std::endl << RESET;
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
    for (auto &input : inputs) {
        auto shape = input.shape();
        std::cout << MAGENTA << "-I-  Input: " << input.name()
        << ", Shape: (" << shape.height << ", " << shape.width << ", " << shape.features << ")"
        << std::endl << RESET;
    }
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
    for (auto &output : outputs) {
        auto shape = output.shape();
        std::cout << MAGENTA << "-I-  Output: " << output.name()
        << ", Shape: (" << shape.height << ", " << shape.width << ", " << shape.features << ")"
        << std::endl << RESET;
    }
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------\n" << std::endl << RESET;
}

void init_video_writer(const std::string &output_path, cv::VideoWriter &video, double fps, int org_width, int org_height) {
    video.open(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(org_width, org_height));
    if (!video.isOpened()) {
        throw std::runtime_error("Error when writing video");
    }
}

cv::VideoCapture open_video_capture(const std::string &input_path, cv::VideoCapture capture,
    double &org_height, double &org_width, size_t &frame_count) {
    capture.open(input_path, cv::CAP_ANY); 
    if (!capture.isOpened()) {
        throw std::runtime_error("Unable to read input file");
    }
    org_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    org_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);
    return capture;
}

bool show_frame(const InputType &input_type, const cv::Mat &frame_to_draw)
{
    if (input_type.is_camera) {
        cv::imshow("Inference", frame_to_draw);
    if (cv::waitKey(1) == 'q') {
        std::cout << "Exiting" << std::endl;
        return false;
        }
    }
    return true;
}

void preprocess_video_frames(cv::VideoCapture &capture,
                          uint32_t width, uint32_t height, size_t batch_size,
                          std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue,
                          PreprocessCallback preprocess_callback) {
    std::vector<cv::Mat> org_frames;
    std::vector<cv::Mat> preprocessed_frames;
    while (true) {
        cv::Mat org_frame;
        capture >> org_frame;
        if (org_frame.empty()) {
            preprocessed_batch_queue->stop();
            break;
        }
        org_frames.push_back(org_frame);
        
        if (org_frames.size() == batch_size) {
            preprocessed_frames.clear();
            preprocess_callback(org_frames, preprocessed_frames, width, height);
            preprocessed_batch_queue->push(std::make_pair(org_frames, preprocessed_frames));
            org_frames.clear();
        }
    }
}

void preprocess_image_frames(const std::string &input_path,
                          uint32_t width, uint32_t height, size_t batch_size,
                          std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue,
                          PreprocessCallback preprocess_callback) {
    cv::Mat org_frame = cv::imread(input_path);
    std::vector<cv::Mat> org_frames = {org_frame}; 
    std::vector<cv::Mat> preprocessed_frames;
    preprocess_callback(org_frames, preprocessed_frames, width, height);
    preprocessed_batch_queue->push(std::make_pair(org_frames, preprocessed_frames));
    
    preprocessed_batch_queue->stop();
}

void preprocess_directory_of_images(const std::string &input_path,
                                uint32_t width, uint32_t height, size_t batch_size,
                                std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue,
                                PreprocessCallback preprocess_callback) {
    std::vector<cv::Mat> org_frames;
    std::vector<cv::Mat> preprocessed_frames;
    
    for (const auto &entry : fs::directory_iterator(input_path)) {
        if (is_image_file(entry.path().string())) {
            cv::Mat org_frame = cv::imread(entry.path().string());
            if (!org_frame.empty()) {
                org_frames.push_back(org_frame);
                
                if (org_frames.size() == batch_size) {
                    preprocessed_frames.clear();
                    preprocess_callback(org_frames, preprocessed_frames, width, height);
                    preprocessed_batch_queue->push(std::make_pair(org_frames, preprocessed_frames));
                    org_frames.clear();
                }
            }
        }
    }    
    preprocessed_batch_queue->stop();
}

hailo_status run_post_process(
    InputType &input_type,
    int org_height,
    int org_width,
    size_t frame_count,
    cv::VideoCapture &capture,
    double fps,
    size_t batch_size,
    std::shared_ptr<BoundedTSQueue<InferenceResult>> results_queue,
    PostprocessCallback postprocess_callback) {
    
    cv::VideoWriter video;
    if (input_type.is_video || input_type.is_camera) {    
        init_video_writer("./processed_video.mp4", video, fps, org_width, org_height);
    }
    int i = 0;
    while (true) {
        show_progress(input_type, i, frame_count);
        InferenceResult output_item;
        if (!results_queue->pop(output_item)) {
            break;
        }
        auto& frame_to_draw = output_item.org_frame;

        if (!output_item.output_data_and_infos.empty() && postprocess_callback) {
            postprocess_callback(frame_to_draw, output_item.output_data_and_infos);
        }
        
        if (input_type.is_video || input_type.is_camera) {
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
    release_resources(capture, video, input_type, nullptr, results_queue);
    return HAILO_SUCCESS;
}

hailo_status run_inference_async(HailoInfer& model,
                            std::chrono::duration<double>& inference_time,
                            std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue,
                            std::shared_ptr<BoundedTSQueue<InferenceResult>> results_queue) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    bool jobs_submitted = false;
    while (true) {
        std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>> preprocessed_frame_items;
        if (!preprocessed_batch_queue->pop(preprocessed_frame_items)) {
            break;
        }
        // Pass as parameters the device input and a lambda that captures the original frame and uses the provided output buffers.
        model.infer(
                    preprocessed_frame_items.second,
                    [org_frames = preprocessed_frame_items.first, queue = results_queue](
                        const hailort::AsyncInferCompletionInfo &info,
                        const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &output_data_and_infos,
                        const std::vector<std::shared_ptr<uint8_t>> &output_guards)
                    {
                        for (size_t i = 0; i < org_frames.size(); ++i) {
                            InferenceResult output_item;
                            output_item.org_frame = org_frames[i];
                            output_item.output_guards.push_back(output_guards[i]);
                            output_item.output_data_and_infos.push_back(output_data_and_infos[i]);
                            queue->push(output_item);
                        }
                    });
        jobs_submitted = true;
    }
    if(jobs_submitted){
        model.wait_for_last_job();
    }
    results_queue->stop();
    auto end_time = std::chrono::high_resolution_clock::now();

    inference_time = end_time - start_time;

    return HAILO_SUCCESS;
}

hailo_status run_preprocess(const std::string& input_path, const std::string& hef_path, HailoInfer &model, 
                            InputType &input_type, cv::VideoCapture &capture,
                            size_t batch_size,
                            std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue,
                            PreprocessCallback preprocess_callback) {

    auto model_input_shape = model.get_infer_model()->hef().get_input_vstream_infos().release()[0].shape;
    print_net_banner(get_hef_name(hef_path), std::ref(model.get_inputs()), std::ref(model.get_outputs()));

    if (input_type.is_image) {
        preprocess_image_frames(input_path, model_input_shape.width, model_input_shape.height, batch_size, preprocessed_batch_queue, preprocess_callback);
    }
    else if (input_type.is_directory) {
        preprocess_directory_of_images(input_path, model_input_shape.width, model_input_shape.height, batch_size, preprocessed_batch_queue, preprocess_callback);
    }
    else{
        preprocess_video_frames(capture, model_input_shape.width, model_input_shape.height, batch_size, preprocessed_batch_queue, preprocess_callback);
    } 
    return HAILO_SUCCESS;
}

void release_resources(cv::VideoCapture &capture, cv::VideoWriter &video, InputType &input_type,
                      std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue,
                      std::shared_ptr<BoundedTSQueue<InferenceResult>> results_queue) {
    if (input_type.is_video) {
        video.release();
    }
    if (input_type.is_camera) {
        capture.release();
        cv::destroyAllWindows();
    }
    if (preprocessed_batch_queue) {
        preprocessed_batch_queue->stop();
    }
    if (results_queue) {
        results_queue->stop();
    }
}
}