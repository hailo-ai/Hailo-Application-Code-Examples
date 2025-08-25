#include "yolov5seg_postprocess.hpp"
#include "toolbox.hpp"
using namespace hailo_utils;

cv::Vec3b indexToColor(size_t index)
{
    return color_table[index % color_table.size()];
}

std::vector<const hailo_detection_with_byte_mask_t*> get_detections(const uint8_t *src_ptr)
{
    std::vector<const hailo_detection_with_byte_mask_t*> detections;
    const uint16_t detections_count = *reinterpret_cast<const uint16_t*>(src_ptr);
    detections.reserve(detections_count);

    size_t offset = sizeof(uint16_t);
    for (size_t i = 0; i < detections_count; ++i) {
        const auto *det_ptr =
            reinterpret_cast<const hailo_detection_with_byte_mask_t*>(src_ptr + offset);
        offset += sizeof(hailo_detection_with_byte_mask_t) + det_ptr->mask_size;
        detections.emplace_back(det_ptr);
    }
    return detections;
}

cv::Mat draw_detections_and_mask(const uint8_t *src_ptr,
                                          int width, int height,
                                          cv::Mat &frame)
{
    auto detections = get_detections(src_ptr);
    cv::Mat overlay = cv::Mat::zeros(height, width, CV_8UC3);
    for (const auto *detection : detections) {
        const int box_w = static_cast<int>(std::ceil((detection->box.x_max - detection->box.x_min) * width));
        const int box_h = static_cast<int>(std::ceil((detection->box.y_max - detection->box.y_min) * height));
        const cv::Vec3b color = indexToColor(detection->class_id);

        const uint8_t *mask_ptr = reinterpret_cast<const uint8_t*>(detection) + sizeof(hailo_detection_with_byte_mask_t);
        const size_t expected = static_cast<size_t>(box_w) * static_cast<size_t>(box_h);
        (void)expected;

        for (int i = 0; i < box_h; ++i) {
            for (int j = 0; j < box_w; ++j) {
                const size_t idx = static_cast<size_t>(i) * static_cast<size_t>(box_w) + static_cast<size_t>(j);
                if (idx < detection->mask_size && mask_ptr[idx]) {
                    const int ox = j + static_cast<int>(detection->box.x_min * width);
                    const int oy = i + static_cast<int>(detection->box.y_min * height);
                    if (0 <= ox && ox < width && 0 <= oy && oy < height) {
                        overlay.at<cv::Vec3b>(oy, ox) = color;
                    }
                }
            }
        }
        cv::rectangle(
            frame,
            cv::Rect(static_cast<int>(detection->box.x_min * width),
                     static_cast<int>(detection->box.y_min * height),
                     box_w, box_h),
            color, 1);
    }

    cv::addWeighted(frame, 1.0, overlay, 0.7, 0.0, frame);
    return frame;
}

cv::Mat pad_frame_letterbox(const cv::Mat &frame, int model_h, int model_w)
{
    // scale to fit (preserve aspect), pad bottom & right
    const float fh = static_cast<float>(frame.rows) / static_cast<float>(model_h);
    const float fw = static_cast<float>(frame.cols) / static_cast<float>(model_w);
    const float factor = std::max(fh, fw); // exactly like your old code

    cv::Mat resized;
    cv::resize(frame, resized,
               cv::Size(static_cast<int>(std::round(frame.cols / factor)),
                        static_cast<int>(std::round(frame.rows / factor))),
               0, 0, cv::INTER_AREA);

    const int pad_h = std::max(0, model_h - resized.rows);
    const int pad_w = std::max(0, model_w - resized.cols);

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, 0, pad_h, 0, pad_w,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return padded;
}

cv::Mat make_model_space_canvas(const cv::Mat &src,
    int model_w, int model_h,
    LetterboxMap &map)
{
const float fh = static_cast<float>(src.rows) / static_cast<float>(model_h);
const float fw = static_cast<float>(src.cols) / static_cast<float>(model_w);
map.factor = std::max(fh, fw);

cv::Mat resized;
cv::resize(src, resized,
cv::Size(static_cast<int>(std::round(src.cols / map.factor)),
static_cast<int>(std::round(src.rows / map.factor))),
0, 0, cv::INTER_AREA);

map.pad_h = std::max(0, model_h - resized.rows);
map.pad_w = std::max(0, model_w - resized.cols);

cv::Mat model_space;
cv::copyMakeBorder(resized, model_space, 0, map.pad_h, 0, map.pad_w,
cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

map.crop_h = static_cast<int>(std::round(src.rows / map.factor));
map.crop_w = static_cast<int>(std::round(src.cols / map.factor));
return model_space;
}

void map_model_to_frame(const cv::Mat &model_space,
    const LetterboxMap &map,
    cv::Mat &dst_frame)
{
const int cw = std::min(map.crop_w, model_space.cols);
const int ch = std::min(map.crop_h, model_space.rows);
const cv::Rect roi(0, 0, cw, ch);

cv::Mat cropped = model_space(roi).clone();
cv::resize(cropped, dst_frame, dst_frame.size(), 0, 0, cv::INTER_LINEAR);
} 