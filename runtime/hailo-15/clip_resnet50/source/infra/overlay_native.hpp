/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
/**
 * @file overlay.hpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-01-20
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#pragma once

#include <map>
#include <vector>
#include "hailo_objects.hpp"
#include "tappas/plugins/common/hailomat.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include "hailo_common.hpp"
typedef enum
{
    OVERLAY_STATUS_UNINITIALIZED = -1,
    OVERLAY_STATUS_OK,

} overlay_status_t;
overlay_status_t draw_all(HailoMat &hmat, HailoROIPtr roi, float landmark_point_radius, bool show_confidence = true, bool local_gallery = false, uint mask_overlay_n_threads = 0);
void face_blur(HailoMat &mat, HailoROIPtr roi);

cv::Scalar indexToColor(size_t index);
#include "tappas/plugins/common/overlay_utils.hpp"

#define SPACE " "
#define TEXT_CLS_FONT_SCALE_FACTOR (0.0025f)
#define MINIMUM_TEXT_CLS_FONT_SCALE (2.2f)
#define TEXT_DEFAULT_HEIGHT (0.05f)
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

static cv::Scalar get_color(size_t color_id)
{
    cv::Scalar color;
    if (NULL_COLOR_ID == color_id)
        color = DEFAULT_COLOR;
    else
        color = indexToColor(color_id);

    return color;
}


std::string confidence_to_string(float confidence)
{
    int confidence_percentage = (confidence * 100);

    return std::to_string(confidence_percentage) + "%";
}

static overlay_status_t draw_classification(HailoMat &mat, HailoROIPtr roi, std::string text, uint number_of_classifications, size_t color_id = NULL_COLOR_ID)
{
    auto bbox = roi->get_bbox();
    int roi_xmin = bbox.xmin() * mat.native_width();
    int roi_ymin = bbox.ymin() * mat.native_height();
    int roi_height = mat.native_height() * bbox.height();
    auto text_position = cv::Point(roi_xmin, roi_ymin + (TEXT_DEFAULT_HEIGHT * number_of_classifications * roi_height) + log(roi_height));
    mat.draw_text(text, text_position, MINIMUM_TEXT_CLS_FONT_SCALE, get_color(color_id));
    return OVERLAY_STATUS_OK;
}

static std::string get_classification_text(HailoClassificationPtr result, bool show_confidence = true)
{
    std::string text;
    std::string label = result->get_label();
    std::string confidence;
    if (show_confidence)
        confidence = SPACE + confidence_to_string(result->get_confidence());
    text = label + confidence;
    return text;
}

static overlay_status_t draw_landmarks(HailoMat &hmat, HailoLandmarksPtr landmarks, HailoROIPtr roi, float landmark_point_radius)
{
    HailoBBox bbox = roi->get_bbox();
    int thickness;
    std::vector<std::pair<int, int>> pairs = landmarks->get_pairs();
    int R = 0;
    std::vector<HailoPoint> points = landmarks->get_points();
    if (landmarks->get_landmarks_type() == "centerpose")
    {
        R = roi->get_bbox().height() * hmat.native_height() / 60;
    }

    for (auto &pair : pairs)
    {
        if ((points.at(pair.first).confidence() > 0) && (points.at(pair.second).confidence() > 0))
        {
            uint x1 = ((points.at(pair.first).x() * bbox.width()) + bbox.xmin()) * hmat.native_width();
            uint y1 = ((points.at(pair.first).y() * bbox.height()) + bbox.ymin()) * hmat.native_height();

            uint x2 = ((points.at(pair.second).x() * bbox.width()) + bbox.xmin()) * hmat.native_width();
            uint y2 = ((points.at(pair.second).y() * bbox.height()) + bbox.ymin()) * hmat.native_height();

            cv::Point joint1 = cv::Point(x1, y1);
            cv::Point joint2 = cv::Point(x2, y2);

            thickness = (bbox.width() < 0.05) ? 1 : 2;
            hmat.draw_line(joint1, joint2, get_color(4), thickness, cv::LINE_4);
        }
    }
    for (auto &point : points)
    {
        if (point.confidence() >= landmarks->get_threshold())
        {
            uint x = ((point.x() * bbox.width()) + bbox.xmin()) * hmat.native_width();
            uint y = ((point.y() * bbox.height()) + bbox.ymin()) * hmat.native_height();
            // Draw the keypoint (multiply x,y values by the sizes of the frame)
            auto center = cv::Point(x, y);
            hmat.draw_ellipse(center, {R, R}, 0, 0, 360, get_color(7), landmark_point_radius);
        }
    }
    return OVERLAY_STATUS_OK;
}

static cv::Rect get_rect(HailoMat &mat, HailoDetectionPtr detection, HailoROIPtr roi)
{
    HailoBBox roi_bbox = hailo_common::create_flattened_bbox(roi->get_bbox(), roi->get_scaling_bbox());
    auto detection_bbox = detection->get_bbox();

    auto bbox_min = cv::Point(((detection_bbox.xmin() * roi_bbox.width()) + roi_bbox.xmin()) * mat.native_width(),
                              ((detection_bbox.ymin() * roi_bbox.height()) + roi_bbox.ymin()) * mat.native_height());
    auto bbox_max = cv::Point(((detection_bbox.xmax() * roi_bbox.width()) + roi_bbox.xmin()) * mat.native_width(),
                              ((detection_bbox.ymax() * roi_bbox.height()) + roi_bbox.ymin()) * mat.native_height());
    return cv::Rect(bbox_min, bbox_max);
}

static std::string get_detection_text(HailoDetectionPtr detection, bool show_confidence = true)
{
    std::string text;
    std::string label = detection->get_label();
    std::string confidence = confidence_to_string(detection->get_confidence());
    if (!show_confidence)
        text = label;
    else if (!label.empty())
    {
        text = label + SPACE + confidence;
    }
    else
    {
        text = confidence;
    }
    return text;
}

static overlay_status_t draw_tile(HailoMat &mat, HailoTileROIPtr tile)
{
    auto bbox = tile->get_bbox();
    auto bbox_min = cv::Point(bbox.xmin() * mat.width(), bbox.ymin() * mat.height());
    auto bbox_max = cv::Point(bbox.xmax() * mat.width(), bbox.ymax() * mat.height());
    cv::Rect rect(bbox_min, bbox_max);
    cv::Scalar color;
    uint tile_layer = tile->get_layer();
    if (tile_layer < tile_layer_color_table.size())
        color = tile_layer_color_table[tile_layer];
    else
        color = get_color(DEFAULT_TILE_COLOR);

    // Draw the tile box
    mat.draw_rectangle(rect, color);

    return OVERLAY_STATUS_OK;
}

static overlay_status_t draw_id(HailoMat &mat, HailoUniqueIDPtr &hailo_id, HailoROIPtr roi)
{
    std::string id_text = std::to_string(hailo_id->get_id());

    auto bbox = roi->get_bbox();
    auto bbox_min = cv::Point(bbox.xmin() * mat.native_width(), bbox.ymin() * mat.native_height());
    auto bbox_max = cv::Point(bbox.xmax() * mat.native_width(), bbox.ymax() * mat.native_height());
    auto bbox_width = bbox_max.x - bbox_min.x;
    auto color = get_color(NULL_CLASS_ID);

    // Calculating the font size according to the box width.
    double font_scale = TEXT_FONT_FACTOR * log(bbox_width);
    auto text_position = cv::Point(bbox_min.x + log(bbox_width), bbox_max.y - log(bbox_width));
    // Draw the class and confidence text
    mat.draw_text(id_text, text_position, font_scale, color);
    return OVERLAY_STATUS_OK;
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
    if (mask->get_height() == 0 || mask->get_width() == 0) {
        return;
    }

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
 * @brief convert the estimated depths to colors and draw it (override the original image), a darker color means that the depth is smaller.
 *
 * @param image_planes: matrix of the image
 * @param mask : HailoDepthMaskPtr that contains the data of the estimated depth of each pixel
 * @param roi region of interest
 * @return overlay_status_t
 */
static overlay_status_t draw_depth_mask(cv::Mat &image_planes, HailoDepthMaskPtr mask, HailoROIPtr roi, const uint mask_overlay_n_threads)
{
    cv::Mat resized_mask_data;
    cv::Mat destinationROI;
    calc_destination_roi_and_resize_mask(destinationROI, image_planes, roi, mask, resized_mask_data, mask->get_data(), CV_32F);

    float min = DEPTH_MIN_DISTANCE;
    float max = DEPTH_MAX_DISTANCE;

    double min_val;
    double max_val;
    cv::Point min_loc;
    cv::Point max_loc;

    cv::minMaxLoc(resized_mask_data, &min_val, &max_val, &min_loc, &max_loc);

    if (max < max_val)
        max = max_val;
    if (min > min_val)
        min = min_val;

    resized_mask_data = (resized_mask_data - min) / (max - min);

    if (mask_overlay_n_threads > 0)
        cv::setNumThreads(mask_overlay_n_threads);

    // perform efficient parallel matrix iteration and color every pixel its class color
    cv::parallel_for_(cv::Range(0, destinationROI.rows * destinationROI.cols), ParallelPixelDepthMask(destinationROI.data, resized_mask_data.data, mask->get_transparency(), image_planes.cols, destinationROI.cols));

    return OVERLAY_STATUS_OK;
}

/**
 * @brief draw a mask whose values are ints represting class ids.
 * draw every pixel in the color in its class color.
 *
 * @param image_planes the image data
 * @param mask  HailoClassMask mask object pointer
 * @param roi the region of interest
 * @return overlay_status_t OVERLAY_STATUS_OK
 */
static overlay_status_t
draw_class_mask(cv::Mat &image_planes, HailoClassMaskPtr mask, HailoROIPtr roi, const uint mask_overlay_n_threads)
{
    cv::Mat resized_mask_data;
    cv::Mat destinationROI;
    calc_destination_roi_and_resize_mask(destinationROI, image_planes, roi, mask, resized_mask_data, mask->get_data(), CV_8UC1);

    if (mask_overlay_n_threads > 0)
        cv::setNumThreads(mask_overlay_n_threads);

    // perform efficient parallel matrix iteration and color every pixel its class color
    cv::parallel_for_(cv::Range(0, destinationROI.rows * destinationROI.cols), ParallelPixelClassMask(destinationROI.data, resized_mask_data.data, mask->get_transparency(), image_planes.cols, destinationROI.cols));

    return OVERLAY_STATUS_OK;
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

    if (mask_overlay_n_threads > 0)
        cv::setNumThreads(mask_overlay_n_threads);

    // perform efficient parallel matrix iteration and color every pixel its class color
    cv::parallel_for_(cv::Range(0, destinationROI.rows * destinationROI.cols), ParallelPixelClassConfMask(destinationROI.data, resized_mask_data.data, mask->get_transparency(), image_planes.cols, destinationROI.cols, mask_color));

    return OVERLAY_STATUS_OK;
}

overlay_status_t draw_all(HailoMat &hmat, HailoROIPtr roi, float landmark_point_radius, bool show_confidence, bool local_gallery, const uint mask_overlay_n_threads)
{
    overlay_status_t ret = OVERLAY_STATUS_UNINITIALIZED;
    uint number_of_classifications = 0;
    cv::Mat &mat = hmat.get_matrices()[0];
    for (auto obj : roi->get_objects())
    {
        switch (obj->get_type())
        {
        case HAILO_DETECTION:
        {
            HailoDetectionPtr detection = std::dynamic_pointer_cast<HailoDetection>(obj);
            if(detection -> get_label() != "face"){
                cv::Scalar color = NO_GLOBAL_ID_COLOR;
                std::string text = "";
                if (local_gallery)
                {
                    auto global_ids = hailo_common::get_hailo_global_id(detection);
                    if (global_ids.size() > 1)
                        std::cerr << "ERROR: more than one global id in roi" << std::endl;
                    if (global_ids.size() == 1)
                        color = GLOBAL_ID_COLOR;
                }
                else
                {
                    color = get_color((size_t)detection->get_class_id());
                    text = get_detection_text(detection, show_confidence);
                }

                // Draw Rectangle
                auto rect = get_rect(hmat, detection, roi);
                hmat.draw_rectangle(rect, color);

                // Draw text
                auto text_position = cv::Point(rect.x - log(rect.width), rect.y - log(rect.width));
                float font_scale = TEXT_FONT_FACTOR * log(rect.width);
                hmat.draw_text(text, text_position, font_scale, color);
                
                // Draw inner objects.
                ret = draw_all(hmat, detection, landmark_point_radius, show_confidence, local_gallery, mask_overlay_n_threads);
            }
            break;
        }
        case HAILO_CLASSIFICATION:
        {
            number_of_classifications++;
            HailoClassificationPtr classification = std::dynamic_pointer_cast<HailoClassification>(obj);
            if (classification->get_classification_type() == "tracking")
            {
                std::string text = get_classification_text(classification, false);
                if (text == "lost")
                    ret = draw_classification(hmat, roi, text, number_of_classifications, 0);
                else if (text == "new")
                    ret = draw_classification(hmat, roi, text, number_of_classifications, 1);
                else if (text == "tracked")
                    ret = draw_classification(hmat, roi, text, number_of_classifications, 2);
            }
            else
            {
                std::string text = get_classification_text(classification, show_confidence);
                ret = draw_classification(hmat, roi, text, number_of_classifications);
            }
            break;
        }
        case HAILO_LANDMARKS:
        {
            HailoLandmarksPtr landmarks = std::dynamic_pointer_cast<HailoLandmarks>(obj);
            draw_landmarks(hmat, landmarks, roi, landmark_point_radius);
            break;
        }
        case HAILO_TILE:
        {
            HailoTileROIPtr tile = std::dynamic_pointer_cast<HailoTileROI>(obj);
            draw_tile(hmat, tile);
            draw_all(hmat, tile, landmark_point_radius, show_confidence, local_gallery, mask_overlay_n_threads);
            break;
        }
        case HAILO_UNIQUE_ID:
        {
            HailoUniqueIDPtr id = std::dynamic_pointer_cast<HailoUniqueID>(obj);
            if ((local_gallery && id->get_mode() == GLOBAL_ID) || (!local_gallery && id->get_mode() == TRACKING_ID))
                draw_id(hmat, id, roi);
            break;
        }
        case HAILO_DEPTH_MASK:
        {
            HailoDepthMaskPtr mask = std::dynamic_pointer_cast<HailoDepthMask>(obj);
            draw_depth_mask(mat, mask, roi, mask_overlay_n_threads);
            break;
        }
        case HAILO_CLASS_MASK:
        {
            HailoClassMaskPtr mask = std::dynamic_pointer_cast<HailoClassMask>(obj);
            draw_class_mask(mat, mask, roi, mask_overlay_n_threads);
            break;
        }
        case HAILO_CONF_CLASS_MASK:
        {
            HailoConfClassMaskPtr mask = std::dynamic_pointer_cast<HailoConfClassMask>(obj);
            draw_conf_class_mask(mat, mask, roi, mask_overlay_n_threads);
            break;
        }
        default:
            // continue
            break;
        }
    }
    ret = OVERLAY_STATUS_OK;
    return ret;
}

void face_blur(HailoMat &hmat, HailoROIPtr roi)
{
    for (auto detection : hailo_common::get_hailo_detections(roi))
    {
        if (detection->get_label() == "face")
        {
            HailoBBox roi_bbox = hailo_common::create_flattened_bbox(roi->get_bbox(), roi->get_scaling_bbox());
            auto detection_bbox = detection->get_bbox();
            auto xmin = std::clamp<int>(((detection_bbox.xmin() * roi_bbox.width()) + roi_bbox.xmin()) * hmat.native_width(), 0, hmat.native_width());
            auto ymin = std::clamp<int>(((detection_bbox.ymin() * roi_bbox.height()) + roi_bbox.ymin()) * hmat.native_height(), 0, hmat.native_height());
            auto xmax = std::clamp<int>(((detection_bbox.xmax() * roi_bbox.width()) + roi_bbox.xmin()) * hmat.native_width(), 0, hmat.native_width());
            auto ymax = std::clamp<int>(((detection_bbox.ymax() * roi_bbox.height()) + roi_bbox.ymin()) * hmat.native_height(), 0, hmat.native_height());
            auto rect = cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax));
            hmat.blur(rect, cv::Size(13, 13));

            // Remove landmarks from the ROI before overlaying the blurred face
            roi->remove_objects_typed(HAILO_LANDMARKS);
        }
        else
        {
            face_blur(hmat, detection);
        }
    }
}
