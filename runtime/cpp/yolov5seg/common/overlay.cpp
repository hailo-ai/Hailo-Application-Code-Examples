/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
/**
 * @file overlay.cpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-01-20
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <opencv2/opencv.hpp>
#include <algorithm>
#include "overlay.hpp"
#include "overlay_utils.hpp"
#include "hailo_common.hpp"

#define SPACE " "
#define TEXT_CLS_FONT_SCALE_FACTOR (0.0025f)
#define MINIMUM_TEXT_CLS_FONT_SCALE (0.5f)
#define TEXT_DEFAULT_HEIGHT (0.1f)
#define TEXT_FONT_FACTOR (0.12f)
#define MINIMAL_BOX_WIDTH_FOR_TEXT (10)
#define LANDMARKS_COLOR (cv::Scalar(255, 0, 0))
#define NO_GLOBAL_ID_COLOR (cv::Scalar(255, 0, 0))
#define GLOBAL_ID_COLOR (cv::Scalar(0, 255, 0))
#define DEFAULT_DETECTION_COLOR (cv::Scalar(255, 255, 255))
#define DEFAULT_TILE_COLOR (2)
#define NULL_COLOR_ID ((size_t)NULL_CLASS_ID)
#define DEFAULT_COLOR (cv::Scalar(255, 255, 255))
// Transformations were taken from https://stackoverflow.com/questions/17892346/how-to-convert-rgb-yuv-rgb-both-ways.
#define RGB2Y(R, G, B) CLIP((0.257 * (R) + 0.504 * (G) + 0.098 * (B)) + 16)
#define RGB2U(R, G, B) CLIP((-0.148 * (R)-0.291 * (G) + 0.439 * (B)) + 128)
#define RGB2V(R, G, B) CLIP((0.439 * (R)-0.368 * (G)-0.071 * (B)) + 128)

#define DEPTH_MIN_DISTANCE 0.5
#define DEPTH_MAX_DISTANCE 3

static const std::vector<cv::Scalar> tile_layer_color_table = {
    cv::Scalar(0, 0, 255), cv::Scalar(200, 100, 120), cv::Scalar(255, 0, 0), cv::Scalar(120, 0, 0), cv::Scalar(0, 0, 120)};

static const std::vector<cv::Scalar> color_table = {
    cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255),
    cv::Scalar(255, 0, 255), cv::Scalar(255, 170, 0), cv::Scalar(255, 0, 170), cv::Scalar(0, 255, 170), cv::Scalar(170, 255, 0),
    cv::Scalar(170, 0, 255), cv::Scalar(0, 170, 255), cv::Scalar(255, 85, 0), cv::Scalar(85, 255, 0), cv::Scalar(0, 255, 85),
    cv::Scalar(0, 85, 255), cv::Scalar(85, 0, 255), cv::Scalar(255, 0, 85), cv::Scalar(255, 255, 255)};


cv::Scalar indexToColor(size_t index)
{
    return color_table[index % color_table.size()];
}

std::string confidence_to_string(float confidence)
{
    int confidence_percentage = (confidence * 100);

    return std::to_string(confidence_percentage) + "%";
}

/**
 * @brief calculate the destionation region of interest and the resized mask
 *
 * @param destinationROI the region of interest to paint
 * @param image_planes the image data
 * @param roi the region of interest
 * @param mask a mask object inherited from from HailoMask
 * @param resized_mask_data an output of the fucntion, the mask resized
 * @param data_ptr mask data pointer
 * @param cv_type type of cv data, example: CV_32F
 */
template <typename T>
void calc_destination_roi_and_resize_mask(cv::Mat &destinationROI, cv::Mat &image_planes, HailoROIPtr roi, HailoMaskPtr mask, cv::Mat &resized_mask_data, T data_ptr, int cv_type)
{
    HailoBBox bbox = roi->get_bbox();
    int roi_xmin = bbox.xmin() * image_planes.cols;
    int roi_ymin = bbox.ymin() * image_planes.rows;
    int roi_width = image_planes.cols * bbox.width();
    int roi_height = image_planes.rows * bbox.height();

    // clamp the region of interest so it is inside the image planes
    roi_xmin = std::clamp(roi_xmin, 0, image_planes.cols);
    roi_ymin = std::clamp(roi_ymin, 0, image_planes.rows);
    roi_width = std::clamp(roi_width, 0, image_planes.cols - roi_xmin);
    roi_height = std::clamp(roi_height, 0, image_planes.rows - roi_ymin);

    cv::Mat mat_data = cv::Mat(mask->get_height(), mask->get_width(), cv_type, (uint8_t *)data_ptr.data());

    cv::resize(mat_data, resized_mask_data, cv::Size(roi_width, roi_height), 0, 0, cv::INTER_LINEAR);

    cv::Rect roi_rect(cv::Point(roi_xmin, roi_ymin), cv::Size(roi_width, roi_height));
    destinationROI = image_planes(roi_rect);
}

/**
 * @brief draw a mask that its values are floats representing confidence.
 * if the pixel value is above threshold, draw this pixel in the mask's class color.
 *
 * @param image_planes the image data
 * @param mask HailoConfClassMask mask object pointer
 * @param roi the region of interest
 * @return overlay_status_t OVERLAY_STATUS_OK
 */
static overlay_status_t draw_conf_class_mask(cv::Mat &image_planes, HailoConfClassMaskPtr mask, HailoROIPtr roi, const uint mask_overlay_n_threads)
{
    cv::Mat resized_mask_data;
    cv::Mat destinationROI;

    calc_destination_roi_and_resize_mask(destinationROI, image_planes, roi, mask, resized_mask_data, mask->get_data(), CV_32F);
    cv::Scalar mask_color = indexToColor(mask->get_class_id());

    // perform efficient parallel matrix iteration and color every pixel its class color
    cv::parallel_for_(cv::Range(0, destinationROI.rows * destinationROI.cols), ParallelPixelClassConfMask(destinationROI.data, resized_mask_data.data, mask->get_transparency(), image_planes.cols, destinationROI.cols, mask_color));

    return OVERLAY_STATUS_OK;
}

overlay_status_t draw_all(cv::Mat &mat, HailoROIPtr roi, const uint mask_overlay_n_threads)
{
    overlay_status_t ret = OVERLAY_STATUS_UNINITIALIZED;
    for (auto& obj : roi->get_objects()) {
        if (obj->get_type() == HAILO_CONF_CLASS_MASK){
            HailoConfClassMaskPtr mask = std::dynamic_pointer_cast<HailoConfClassMask>(obj);
            if (mask->get_height() != 0 && mask->get_width() != 0)
                draw_conf_class_mask(mat, mask, roi, mask_overlay_n_threads);
        }
    }
    ret = OVERLAY_STATUS_OK;
    return ret;
}