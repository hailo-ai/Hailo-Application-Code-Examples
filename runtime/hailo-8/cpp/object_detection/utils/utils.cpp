#include "utils.hpp"

std::vector<cv::Scalar> COLORS = {
    cv::Scalar(255,   0,   0),  // Red
    cv::Scalar(  0, 255,   0),  // Green
    cv::Scalar(  0,   0, 255),  // Blue
    cv::Scalar(255, 255,   0),  // Cyan
    cv::Scalar(255,   0, 255),  // Magenta
    cv::Scalar(  0, 255, 255),  // Yellow
    cv::Scalar(255, 128,   0),  // Orange
    cv::Scalar(128,   0, 128),  // Purple
    cv::Scalar(128, 128,   0),  // Olive
    cv::Scalar(128,   0, 255),  // Violet
    cv::Scalar(  0, 128, 255),  // Sky Blue
    cv::Scalar(255,   0, 128),  // Pink
    cv::Scalar(  0, 128,   0),  // Dark Green
    cv::Scalar(128, 128, 128),  // Gray
    cv::Scalar(255, 255, 255)   // White
};


std::string get_coco_name_from_int(int cls)
{
    std::string result = "N/A";
    switch(cls) {
        case 0:  result = "__background__";   break;
        case 1:  result = "person";           break;
        case 2:  result = "bicycle";          break;
        case 3:  result = "car";              break;
        case 4:  result = "motorcycle";       break;
        case 5:  result = "airplane";         break;
        case 6:  result = "bus";              break;
        case 7:  result = "train";            break;
        case 8:  result = "truck";            break;
        case 9:  result = "boat";             break;
        case 10: result = "traffic light";    break;
        case 11: result = "fire hydrant";     break;
        case 12: result = "stop sign";        break;
        case 13: result = "parking meter";    break;
        case 14: result = "bench";            break;
        case 15: result = "bird";             break;
        case 16: result = "cat";              break;
        case 17: result = "dog";              break;
        case 18: result = "horse";            break;
        case 19: result = "sheep";            break;
        case 20: result = "cow";              break;
        case 21: result = "elephant";         break;
        case 22: result = "bear";             break;
        case 23: result = "zebra";            break;
        case 24: result = "giraffe";          break;
        case 25: result = "backpack";         break;
        case 26: result = "umbrella";         break;
        case 27: result = "handbag";          break;
        case 28: result = "tie";              break;
        case 29: result = "suitcase";         break;
        case 30: result = "frisbee";          break;
        case 31: result = "skis";             break;
        case 32: result = "snowboard";        break;
        case 33: result = "sports ball";      break;
        case 34: result = "kite";             break;
        case 35: result = "baseball bat";     break;
        case 36: result = "baseball glove";   break;
        case 37: result = "skateboard";       break;
        case 38: result = "surfboard";        break;
        case 39: result = "tennis racket";    break;
        case 40: result = "bottle";           break;
        case 41: result = "wine glass";       break;
        case 42: result = "cup";              break;
        case 43: result = "fork";             break;
        case 44: result = "knife";            break;
        case 45: result = "spoon";            break;
        case 46: result = "bowl";             break;
        case 47: result = "banana";           break;
        case 48: result = "apple";            break;
        case 49: result = "sandwich";         break;
        case 50: result = "orange";           break;
        case 51: result = "broccoli";         break;
        case 52: result = "carrot";           break;
        case 53: result = "hot dog";          break;
        case 54: result = "pizza";            break;
        case 55: result = "donut";            break;
        case 56: result = "cake";             break;
        case 57: result = "chair";            break;
        case 58: result = "couch";            break;
        case 59: result = "potted plant";     break;
        case 60: result = "bed";              break;
        case 61: result = "dining table";     break;
        case 62: result = "toilet";           break;
        case 63: result = "tv";               break;
        case 64: result = "laptop";           break;
        case 65: result = "mouse";            break;
        case 66: result = "remote";           break;
        case 67: result = "keyboard";         break;
        case 68: result = "cell phone";       break;
        case 69: result = "microwave";        break;
        case 70: result = "oven";             break;
        case 71: result = "toaster";          break;
        case 72: result = "sink";             break;
        case 73: result = "refrigerator";     break;
        case 74: result = "book";             break;
        case 75: result = "clock";            break;
        case 76: result = "vase";             break;
        case 77: result = "scissors";         break;
        case 78: result = "teddy bear";       break;
        case 79: result = "hair drier";       break;
        case 80: result = "toothbrush";       break;
    }
    return result;
}


CommandLineArgs parse_command_line_arguments(int argc, char** argv) {
    return {
        getCmdOption(argc, argv, "-hef="),
        getCmdOption(argc, argv, "-input="),
        has_flag(argc, argv, "-s")
    };
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

bool is_directory_of_images(const std::string& path, int &entry_count) {
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

InputType determine_input_type(const std::string& input_path, cv::VideoCapture &capture,
                               double &org_height, double &org_width, size_t &frame_count) {

    InputType input_type;
    int directory_entry_count;
    if (is_directory_of_images(input_path, directory_entry_count)) {
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

PreprocessedFrameItem create_preprocessed_frame_item(const cv::Mat &frame,
                                                            uint32_t width,
                                                            uint32_t height)
{
    PreprocessedFrameItem item;
    item.org_frame = frame.clone(); 
    cv::resize(frame, item.resized_for_infer, cv::Size(width, height));
    return item;
}

void initialize_class_colors(std::unordered_map<int, cv::Scalar>& class_colors) {
    for (int cls = 0; cls <= 80; ++cls) {
        class_colors[cls] = COLORS[cls % COLORS.size()]; 
    }
}

std::string get_hef_name(const std::string &path)
{
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
}

bool has_flag(int argc, char *argv[], const std::string &flag) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == flag) {
            return true;
        }
    }
    return false;
}

hailo_status check_status(const hailo_status &status, const std::string &message) {
    if (HAILO_SUCCESS != status) {
        std::cerr << message << " with status " << status << std::endl;
        return status;
    }
    return HAILO_SUCCESS;
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

void show_progress(InputType &input_type, int progress, size_t frame_count) {
    if (input_type.is_video) {
        show_progress_helper(progress, frame_count);
    } else if (input_type.is_directory) {
        show_progress_helper(progress, input_type.directory_entry_count);
    }
}

cv::Rect get_bbox_coordinates(const hailo_bbox_float32_t& bbox, int frame_width, int frame_height) {
    int x1 = static_cast<int>(bbox.x_min * frame_width);
    int y1 = static_cast<int>(bbox.y_min * frame_height);
    int x2 = static_cast<int>(bbox.x_max * frame_width);
    int y2 = static_cast<int>(bbox.y_max * frame_height);
    return cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
}

void draw_label(cv::Mat& frame, const std::string& label, const cv::Point& top_left, const cv::Scalar& color) {
    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    int top = std::max(top_left.y, label_size.height);
    cv::rectangle(frame, cv::Point(top_left.x, top + baseLine), 
                  cv::Point(top_left.x + label_size.width, top - label_size.height), color, cv::FILLED);
    cv::putText(frame, label, cv::Point(top_left.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
}

void draw_single_bbox(cv::Mat& frame, const NamedBbox& named_bbox, const cv::Scalar& color) {
    auto bbox_rect = get_bbox_coordinates(named_bbox.bbox, frame.cols, frame.rows);
    cv::rectangle(frame, bbox_rect, color, 2);

    std::string score_str = std::to_string(named_bbox.bbox.score * 100).substr(0, 4) + "%";
    std::string label = get_coco_name_from_int(static_cast<int>(named_bbox.class_id)) + " " + score_str;
    draw_label(frame, label, bbox_rect.tl(), color);
}

void draw_bounding_boxes(cv::Mat& frame, const std::vector<NamedBbox>& bboxes) {
    std::unordered_map<int, cv::Scalar> class_colors;
    initialize_class_colors(class_colors);
    for (const auto& named_bbox : bboxes) {
        const auto& color = class_colors[named_bbox.class_id];
        draw_single_bbox(frame, named_bbox, color);
    }
}

std::vector<NamedBbox> parse_nms_data(uint8_t* data, size_t max_class_count) {
    std::vector<NamedBbox> bboxes;
    size_t offset = 0;

    for (size_t class_id = 0; class_id < max_class_count; class_id++) {
        auto det_count = static_cast<uint32_t>(*reinterpret_cast<float32_t*>(data + offset));
        offset += sizeof(float32_t);

        for (size_t j = 0; j < det_count; j++) {
            hailo_bbox_float32_t bbox_data = *reinterpret_cast<hailo_bbox_float32_t*>(data + offset);
            offset += sizeof(hailo_bbox_float32_t);

            NamedBbox named_bbox;
            named_bbox.bbox = bbox_data;
            named_bbox.class_id = class_id + 1;
            bboxes.push_back(named_bbox);
        }
    }
    return bboxes;
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
