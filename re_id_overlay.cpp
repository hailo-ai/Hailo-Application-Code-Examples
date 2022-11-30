/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/

// #include <gst/video/video-format.h>
#include <iostream>
#include <map>
#include <typeinfo>
#include <math.h>

// Hailo includes
#include "re_id_overlay.hpp"
#include "hailo_common.hpp"

// Open source includes
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

// General
#define TEXT_THICKNESS (1)
#define TEXT_CLS_THICKNESS (2)
#define TEXT_FONT_FACTOR (0.12f)
#define DEFAULT_DETECTION_COLOR (cv::Scalar(255, 255, 255))

static const std::vector<cv::Scalar> color_table = {
    cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255),
    cv::Scalar(255, 0, 255), cv::Scalar(255, 170, 0), cv::Scalar(255, 0, 170), cv::Scalar(0, 255, 170), cv::Scalar(170, 255, 0),
    cv::Scalar(170, 0, 255), cv::Scalar(0, 170, 255), cv::Scalar(255, 85, 0), cv::Scalar(85, 255, 0), cv::Scalar(0, 255, 85),
    cv::Scalar(0, 85, 255), cv::Scalar(85, 0, 255), cv::Scalar(255, 0, 85)};

cv::Scalar indexToColor(size_t index)
{
    return color_table[index % color_table.size()];
}

HailoUniqueIDPtr get_global_id(HailoDetectionPtr detection)
{
    for (auto obj : detection->get_objects_typed(HAILO_UNIQUE_ID))
    {
        HailoUniqueIDPtr id = std::dynamic_pointer_cast<HailoUniqueID>(obj);
        if (id->get_mode() == GLOBAL_ID)
        {
            return id;
        }
    }
    return nullptr;
}

void face_blur(cv::Mat &mat, HailoROIPtr roi)
{
    for (auto detection : hailo_common::get_hailo_detections(roi))
    {
        if (detection->get_label() == "face")
        {
            HailoBBox roi_bbox = hailo_common::create_flattened_bbox(roi->get_bbox(), roi->get_scaling_bbox());
            auto detection_bbox = detection->get_bbox();
            auto xmin = std::clamp<int>(((detection_bbox.xmin() * roi_bbox.width()) + roi_bbox.xmin()) * mat.cols, 0, mat.cols);
            auto ymin = std::clamp<int>(((detection_bbox.ymin() * roi_bbox.height()) + roi_bbox.ymin()) * mat.rows, 0, mat.rows);
            auto xmax = std::clamp<int>(((detection_bbox.xmax() * roi_bbox.width()) + roi_bbox.xmin()) * mat.cols, 0, mat.cols);
            auto ymax = std::clamp<int>(((detection_bbox.ymax() * roi_bbox.height()) + roi_bbox.ymin()) * mat.rows, 0, mat.rows);
            auto bbox_min = cv::Point(xmin, ymin);
            auto bbox_max = cv::Point(xmax, ymax);
            auto rect = cv::Rect(bbox_min, bbox_max);
            cv::Mat face = mat(rect);
            cv::blur(face, face, cv::Size(11, 11));
            cv::rectangle(mat, bbox_min, bbox_max, cv::Scalar(0,0,0), 2);
            roi->remove_object(detection);
        }
        else
        {
            face_blur(mat, detection);
        }
    }
}

static void draw_detection(cv::Mat &image_planes, HailoDetectionPtr detection, HailoROIPtr roi, int font_thickness = 1, int line_thickness = 1)
{

    HailoBBox roi_bbox = hailo_common::create_flattened_bbox(roi->get_bbox(), roi->get_scaling_bbox());
    auto detection_bbox = detection->get_bbox();
    auto global_id = get_global_id(detection);

    auto bbox_min = cv::Point(((detection_bbox.xmin() * roi_bbox.width()) + roi_bbox.xmin()) * image_planes.cols,
                              ((detection_bbox.ymin() * roi_bbox.height()) + roi_bbox.ymin()) * image_planes.rows);
    auto bbox_max = cv::Point(((detection_bbox.xmax() * roi_bbox.width()) + roi_bbox.xmin()) * image_planes.cols,
                              ((detection_bbox.ymax() * roi_bbox.height()) + roi_bbox.ymin()) * image_planes.rows);
    auto bbox_width = bbox_max.x - bbox_min.x;

    // Draw the detection box
    auto color_rgb = (!global_id) ? DEFAULT_DETECTION_COLOR : indexToColor(global_id->get_id());
    cv::rectangle(image_planes, bbox_min, bbox_max, color_rgb, line_thickness);

    if (global_id)
    {
        std::string id_text = std::to_string(global_id->get_id());
        // Calculating the font size according to the box width.
        float font_scale = TEXT_FONT_FACTOR * log(bbox_width);
        auto text_position = cv::Point(bbox_min.x + log(bbox_width), bbox_max.y - log(bbox_width));
        // Draw the class and confidence text
        cv::putText(image_planes, id_text, text_position, cv::FONT_HERSHEY_SIMPLEX, font_scale, color_rgb, font_thickness);
    }
}

// void filter(HailoROIPtr roi, GstVideoFrame *frame, gchar *current_stream_id)
void filter(HailoROIPtr roi, cv::Mat& frame)
{
    // gint cv2_format = CV_8UC3;
    int font_thickness = 2;
    int line_thickness = 2;
    // guint matrix_width = (guint)GST_VIDEO_FRAME_WIDTH(frame);

    // auto mat = cv::Mat(GST_VIDEO_FRAME_HEIGHT(frame), matrix_width, cv2_format,
    //                    GST_VIDEO_FRAME_PLANE_DATA(frame, 0), GST_VIDEO_FRAME_PLANE_STRIDE(frame, 0));
    face_blur(frame, roi);

    for (auto obj : roi->get_objects())
    {
        switch (obj->get_type())
        {
        case HAILO_DETECTION:
        {

            HailoDetectionPtr detection = std::dynamic_pointer_cast<HailoDetection>(obj);
            draw_detection(frame, detection, roi, font_thickness, line_thickness);
            break;
        }
        default:
            // continue
            break;
        }
    }

    // mat.release();
}