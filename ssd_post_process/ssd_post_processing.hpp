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

#ifndef _HAILO_SSD_POST_PROCESSING_HPP_
#define _HAILO_SSD_POST_PROCESSING_HPP_

#include <vector>
#include <unordered_map>
#include <stdint.h>
#include <memory>
#include <iostream>
#include "common.h"

// === CONFIGURATION =======================================================================================
#define IMAGE_SIZE           300
#define CONFIDENCE_THRESHOLD 0.4f
#define IOU_THRESHOLD        0.6f
#define MAX_BOXES            50
const std::vector<float> BOX_CODER_SCALE = {10.0, 10.0, 5.0, 5.0};
// =========================================================================================================


struct DetectionObject {
    float ymin, xmin, ymax, xmax, confidence;
    uint32_t class_id;

    DetectionObject(float ymin, float xmin, float ymax, float xmax, float confidence, int class_id):
        ymin(ymin), xmin(xmin), ymax(ymax), xmax(xmax), confidence(confidence), class_id(class_id)
        {}

    bool operator<(const DetectionObject &s2) const {
        return this->confidence > s2.confidence;
    }

    friend std::ostream& operator<<(std::ostream& os, const DetectionObject& d) {
        os << "cls: " << get_coco17_name_from_int(d.class_id) << ", box: " << d.ymin << " " << d.xmin << " " << d.ymax << " " << d.xmax << " , conf:" << d.confidence;
        return os;
    }
};

class OutTensor {
public:
    uint8_t* m_data;
    float qp_zp;
    float qp_scale;
    int height;
    int width;
    int channels;

    OutTensor() : m_data(nullptr), qp_zp(-1.f), qp_scale(-1.f), height(-1.f), width(-1.f), channels(-1.f)
    {}

    OutTensor(float qp_zp, float qp_scale, int height, int width, int channels) : m_data(nullptr), qp_zp(qp_zp), qp_scale(qp_scale), height(height), width(width), channels(channels)
    {}

    OutTensor(uint8_t* m_data, float qp_zp, float qp_scale, int height, int width, int channels) : m_data(m_data), qp_zp(qp_zp), qp_scale(qp_scale), height(height), width(width), channels(channels)
    {}

    friend std::ostream& operator<<(std::ostream& os, const OutTensor& t) {
        os << MAGENTA << "OutTensor: h " << t.height << ", w " << t.width << ", c " << t.channels << RESET;
        return os;
    }
};


std::vector<DetectionObject> post_processing(std::vector<std::pair<OutTensor,OutTensor>> &tensors);

#endif /* _HAILO_SSD_POST_PROCESSING_HPP_ */
