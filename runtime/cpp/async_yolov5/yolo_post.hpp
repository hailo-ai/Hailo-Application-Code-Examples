/**
 * Copyright 2023 (C) Hailo Technologies Ltd.
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

#ifndef _HAILO_YOLO_POST_HPP_
#define _HAILO_YOLO_POST_HPP_

#include "hailo/hailort.hpp"
#include <algorithm>
#include <vector>
#include <memory>

constexpr float default_conf_threshold = 0.6f;
constexpr int default_anchors_num = 3;
constexpr int default_feature_map_channels = 85;

// ------------- in note for debug only ! -------------------------------------------------------------------
struct DetectionObject {
    float ymin, xmin, ymax, xmax, confidence;
    int class_id;

    DetectionObject(float ymin, float xmin, float ymax, float xmax, float confidence, int class_id):
        ymin(ymin), xmin(xmin), ymax(ymax), xmax(xmax), confidence(confidence), class_id(class_id)
        {}

    DetectionObject() : ymin(0.), xmin(0.), ymax(1.), xmax(1.), confidence(0.), class_id(0.) {}

    bool operator<(const DetectionObject &s2) const {
        return this->confidence > s2.confidence;
    }
};

class FeatureMap {
public:
    FeatureMap() : data(nullptr), height(0), width(0), channels(0), m_qp_zp(0.), m_qp_scale(1.), conf_threshold(default_conf_threshold), anchors_num(default_anchors_num), feature_map_channels(default_feature_map_channels) {}
    FeatureMap(std::shared_ptr<uint8_t> data, int height, int width, int channels, int anchors_num, int feature_map_channels, float32_t m_qp_zp, float32_t m_qp_scale, float conf_threshold, std::vector<int> anchors) :
        data(data), height(height), width(width), channels(channels), anchors_num(anchors_num), feature_map_channels(feature_map_channels), m_qp_zp(m_qp_zp), m_qp_scale(m_qp_scale), conf_threshold(conf_threshold), anchors(anchors) {}
    void extract_boxes(std::vector<DetectionObject>& detections, const int max_num_detections);

private:
    std::shared_ptr<uint8_t> data;
    int height;
    int width;
    int channels;
    const int anchors_num;
    const int feature_map_channels;
    float32_t m_qp_zp;
    float32_t m_qp_scale;
    float conf_threshold;
    std::vector<int> anchors;

};

class YoloPost {
public:
    YoloPost();
    YoloPost(float conf_threshold, float iou_threshold, int max_num_detections);

    void iou_over_frame();
    const std::vector<DetectionObject> get_detections() const;
    std::vector<DetectionObject> decode();

// private:
    std::vector<DetectionObject> detections;
    int num_detections;
    int max_num_detections;
    float conf_threshold;
    float iou_threshold;
    int num_outputs;
    std::vector<FeatureMap> feature_maps;

};

#endif /* _HAILO_YOLOV5_POST_PROCESSING_HPP_ */
