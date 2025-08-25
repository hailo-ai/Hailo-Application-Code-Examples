/**
 * Copyright 2020 (C) Hailo Technologies Ltd.
 * All rights reserved.
 *
 * Hailo Technologies Ltd. ("Hailo") disclaims any warranties, including, but not limited to,
 * the implied warranties of merchantability and fitness for a particular purpose.
 * This software is provided on an "AS IS" basis, and Hailo has no obligation to provide maintenance,
 * support, updates, enhancements, or modifications.
 *
 * You may use this software in the development of any project.
 * You shall not reproduce, modify or distribute this software without prior written permission.
 **/

/**
 * @file yolov5seg_postprocess.hpp
 * YOLOv5 Segmentation post-processing utilities
 **/

#ifndef _YOLOV5SEG_POSTPROCESS_H_
#define _YOLOV5SEG_POSTPROCESS_H_

#include <iostream>
#include <string>
#include <vector>
#include "hailo/hailort.h"
#include "toolbox.hpp"
#pragma once
#include <opencv2/opencv.hpp>
using namespace hailo_utils;
// --------------------------- FUNCTION DECLARATIONS ---------------------------

static const std::vector<cv::Vec3b> color_table = {
    cv::Vec3b(255, 0, 0), cv::Vec3b(0, 255, 0), cv::Vec3b(0, 0, 255), cv::Vec3b(255, 255, 0), cv::Vec3b(0, 255, 255),
    cv::Vec3b(255, 0, 255), cv::Vec3b(255, 170, 0), cv::Vec3b(255, 0, 170), cv::Vec3b(0, 255, 170), cv::Vec3b(170, 255, 0),
    cv::Vec3b(170, 0, 255), cv::Vec3b(0, 170, 255), cv::Vec3b(255, 85, 0), cv::Vec3b(85, 255, 0), cv::Vec3b(0, 255, 85),
    cv::Vec3b(0, 85, 255), cv::Vec3b(85, 0, 255), cv::Vec3b(255, 0, 85)};

struct LetterboxMap {
    float factor{1.0f};
    int pad_w{0};
    int pad_h{0};
    int crop_w{0};
    int crop_h{0};
};
    
// ─────────────────────────────────────────────────────────────────────────────
// COMMAND-LINE PARSING / FLAGS
// ─────────────────────────────────────────────────────────────────────────────

std::string getCmdOptionWithShortFlag(int argc, char *argv[], const std::string &longOption, const std::string &shortOption);


template <typename T>
cv::Mat draw_detections_and_mask(std::vector<T>& logits,
                                 int width, int height,
                                 cv::Mat& frame);

                                 
std::vector<const hailo_detection_with_byte_mask_t*> get_detections(const uint8_t *src_ptr);

cv::Mat draw_detections_and_mask(const uint8_t *src_ptr,
                                          int width, int height,
                                          cv::Mat &frame);

cv::Mat pad_frame_letterbox(const cv::Mat &frame, int model_h, int model_w);

cv::Mat make_model_space_canvas(const cv::Mat &src,
    int model_w, int model_h,
    LetterboxMap &map);

void map_model_to_frame(const cv::Mat &model_space,
    const LetterboxMap &map,
    cv::Mat &dst_frame);


#endif // _YOLOV5SEG_POSTPROCESS_H_ 