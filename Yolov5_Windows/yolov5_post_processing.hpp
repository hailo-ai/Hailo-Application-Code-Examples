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
 * @file yolov5_post_processing.hpp
 * @brief Yolov5 Post-Processing
 **/

#ifndef _HAILO_YOLOV5_POST_PROCESSING_HPP_
#define _HAILO_YOLOV5_POST_PROCESSING_HPP_




#include <vector>
#include <unordered_map>
#include <stdint.h>

//constexpr uint32_t IMAGE_SIZE = 640;
constexpr uint32_t YOLOV5M_IMAGE_WIDTH = 640;
constexpr uint32_t YOLOV5M_IMAGE_HEIGHT = 640;
//const uint32_t MAX_BOXES = 50;

struct DetectionObject {
    float ymin, xmin, ymax, xmax, confidence;
    uint32_t class_id;

    DetectionObject(float ymin, float xmin, float ymax, float xmax, float confidence, uint32_t class_id):
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
    uint8_t *fm1, float qp_zp_1, float qp_scale_1,
    uint8_t *fm2, float qp_zp_2, float qp_scale_2,
    uint8_t *fm3, float qp_zp_3, float qp_scale_3);

static std::unordered_map<uint32_t, std::string> coco_eighty_classes = {
        {0, "unlabeled"},
        {1, "person"},
        {10, "trafficlight"},
        {11, "firehydrant"},
        {12, "stopsign"},
        {13, "parkingmeter"},
        {14, "bench"},
        {15, "bird"},
        {16, "cat"},
        {17, "dog"},
        {18, "horse"},
        {19, "sheep"},
        {2, "bicycle"},
        {20, "cow"},
        {21, "elephant"},
        {22, "bear"},
        {23, "zebra"},
        {24, "giraffe"},
        {25, "backpack"},
        {26, "umbrella"},
        {27, "handbag"},
        {28, "tie"},
        {29, "suitcase"},
        {3, "car"},
        {30, "frisbee"},
        {31, "skis"},
        {32, "snowboard"},
        {33, "sportsball"},
        {34, "kite"},
        {35, "baseballbat"},
        {36, "baseballglove"},
        {37, "skateboard"},
        {38, "surfboard"},
        {39, "tennisracket"},
        {4, "motorcycle"},
        {40, "bottle"},
        {41, "wineglass"},
        {42, "cup"},
        {43, "fork"},
        {44, "knife"},
        {45, "spoon"},
        {46, "bowl"},
        {47, "banana"},
        {48, "apple"},
        {49, "sandwich"},
        {5, "airplane"},
        {50, "orange"},
        {51, "broccoli"},
        {52, "carrot"},
        {53, "hotdog"},
        {54, "pizza"},
        {55, "donut"},
        {56, "cake"},
        {57, "chair"},
        {58, "couch"},
        {59, "pottedplant"},
        {60, "bed"},
        {61, "diningtable"},
        {62, "toilet"},
        {63, "tv"},
        {64, "laptop"},
        {65, "mouse"},
        {66, "remote"},
        {67, "keyboard"},
        {68, "cellphone"},
        {69, "microwave"},
        {7, "train"},
        {70, "oven"},
        {71, "toaster"},
        {72, "sink"},
        {73, "refrigerator"},
        {74, "book"},
        {75, "clock"},
        {76, "vase"},
        {77, "scissors"},
        {78, "teddybear"},
        {79, "hairdrier"},
        {8, "truck"},
        {80, "toothbrush"},
        {9, "boat"}};

#endif /* _HAILO_YOLOV5_POST_PROCESSING_HPP_ */
