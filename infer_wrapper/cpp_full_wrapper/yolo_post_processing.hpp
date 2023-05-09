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
 * @file yolo_post_processing.hpp
 * @brief Yolo Post-Processing
 **/

#ifndef _HAILO_YOLOV5_POST_PROCESSING_HPP_
#define _HAILO_YOLOV5_POST_PROCESSING_HPP_

#include <vector>
#include <unordered_map>
#include <stdint.h>

typedef float float32_t;

struct DetectionObject {
    float ymin, xmin, ymax, xmax, confidence;
    uint32_t class_id;

    DetectionObject(float ymin, float xmin, float ymax, float xmax, float confidence, int class_id):
        ymin(ymin), xmin(xmin), ymax(ymax), xmax(xmax), confidence(confidence), class_id(class_id)
        {}

    bool operator<(const DetectionObject &s2) const {
        return this->confidence > s2.confidence;
    }
};

/*
    API for yolo postprocessing
    Given all parameters this function returns boxes with class and confidence
    Inputs:
        feature map1: 20x20x255
        feature map2: 40x40x255
        feature map3: 80x80x255
    Outputs:
        final boxes for display (Nx6) - DetectionObject
*/
std::vector<DetectionObject> post_processing(
    int max_num_detections, float thr, std::string &arch,
    uint8_t *fm1, float qp_zp_1, float qp_scale_1,
    uint8_t *fm2, float qp_zp_2, float qp_scale_2,
    uint8_t *fm3, float qp_zp_3, float qp_scale_3);

#endif /* _HAILO_YOLOV5_POST_PROCESSING_HPP_ */
