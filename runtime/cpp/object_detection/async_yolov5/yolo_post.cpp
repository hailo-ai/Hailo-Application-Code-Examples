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
 * @file yolo_v5_post.cpp
 * @brief Yolov5 Post-Processing
 **/

#include "yolo_post.hpp"
#include "hailo/hailort.hpp"

#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>

#define YOLOV5M_IMAGE_SIZE   640
#define CONF_CHANNEL_OFFSET  4
#define CLASS_CHANNEL_OFFSET 5


// TODO: in hpp file

float iou(const DetectionObject& box_1, const DetectionObject& box_2) {
    const float width_of_overlap_area = std::min(box_1.xmax, box_2.xmax) - std::max(box_1.xmin, box_2.xmin);
    const float height_of_overlap_area = std::min(box_1.ymax, box_2.ymax) - std::max(box_1.ymin, box_2.ymin);
    const float positive_width_of_overlap_area = std::max(width_of_overlap_area, 0.0f);
    const float positive_height_of_overlap_area = std::max(height_of_overlap_area, 0.0f);
    const float area_of_overlap = positive_width_of_overlap_area * positive_height_of_overlap_area;
    const float box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin);
    const float box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin);
    return area_of_overlap / (box_1_area + box_2_area - area_of_overlap);
}

float fix_scale(uint8_t& input, float &qp_scale, float &qp_zp) {
  return (float(input) - qp_zp) * qp_scale;
}

YoloPost::YoloPost() : num_detections(0), conf_threshold(default_conf_threshold), iou_threshold(default_iou_threshold), num_outputs(default_num_outputs), max_num_detections(default_max_num_detections) {
    feature_maps.reserve(num_outputs);
    detections.reserve(default_max_num_detections);
}

YoloPost::YoloPost(float conf_threshold, float iou_threshold, int max_num_detections)
    : num_detections(0), conf_threshold(conf_threshold), iou_threshold(iou_threshold), num_outputs(default_num_outputs), max_num_detections(max_num_detections) {
    feature_maps.reserve(num_outputs);
    detections.reserve(max_num_detections);
}

void FeatureMap::extract_boxes(std::vector<DetectionObject>& detections, const int max_num_detections) {
    float  confidence, x, y, h, w, xmin, ymin, xmax, ymax, conf_max = 0.0f;
    int add = 0, anchor = 0, chosen_row = 0, chosen_col = 0, chosen_cls = -1;
    uint8_t cls_prob, prob_max;
    // channels 0-3 are box coordinates, channel 4 is the confidence, and channels 5-84 are classes

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            prob_max = 0;
            for (int a = 0; a < anchors_num; ++a) {
                add =  (feature_map_channels * anchors_num + 1) * width * row + (feature_map_channels * anchors_num + 1) * col + feature_map_channels * a + CONF_CHANNEL_OFFSET;
                confidence = fix_scale(data.get()[add], m_qp_scale, m_qp_zp);
                if (confidence < conf_threshold)
                    continue;
                for (int c = CLASS_CHANNEL_OFFSET; c < feature_map_channels; ++c) {
                    add = (feature_map_channels * anchors_num + 1) * width * row + (feature_map_channels * anchors_num + 1) * col + feature_map_channels * a + c;
                    // final confidence: box confidence * class probability
                    cls_prob = data.get()[add];
                    if (cls_prob > prob_max) 
					{
                        conf_max = fix_scale(cls_prob, m_qp_scale,  m_qp_zp) * confidence;
                        chosen_cls = c - CLASS_CHANNEL_OFFSET + 1;
                        prob_max = cls_prob;
                        anchor = a;
                        chosen_row = row;
                        chosen_col = col;
                    }
                }
                if (conf_max >= conf_threshold) {
                    add = (feature_map_channels * anchors_num + 1) * width * chosen_row + (feature_map_channels * anchors_num + 1) * chosen_col + feature_map_channels * anchor;
                    // box centers
                    x = (fix_scale(data.get()[add], m_qp_scale,  m_qp_zp) * 2.0f - 0.5f + (float)(chosen_col)) / ((float)(width));
                    y = (fix_scale(data.get()[add + 1], m_qp_scale,  m_qp_zp) * 2.0f - 0.5f +  (float)(chosen_row)) / (float)(height);
                    // box scales
                    w = (float)pow(2.0f * (fix_scale(data.get()[add + 2], m_qp_scale,  m_qp_zp)), 2.0f) * (float)(anchors[anchor * 2]) / YOLOV5M_IMAGE_SIZE;
                    h = (float)pow(2.0f * (fix_scale(data.get()[add + 3], m_qp_scale,  m_qp_zp)), 2.0f) * (float)(anchors[anchor * 2 + 1]) / YOLOV5M_IMAGE_SIZE;
                    // x,y,h,w to xmin,ymin,xmax,ymax
                    xmin = std::max(((x - (w / 2.0f)) * YOLOV5M_IMAGE_SIZE), 0.0f);
                    ymin = std::max(((y - (h / 2.0f)) * YOLOV5M_IMAGE_SIZE), 0.0f);
                    xmax = std::min(((x + (w / 2.0f)) * YOLOV5M_IMAGE_SIZE), (static_cast<float>(YOLOV5M_IMAGE_SIZE) - 1));
                    ymax = std::min(((y + (h / 2.0f)) * YOLOV5M_IMAGE_SIZE), (static_cast<float>(YOLOV5M_IMAGE_SIZE) - 1));

                    if (detections.size() >= max_num_detections) {
                        return;
					}
                    detections.push_back(DetectionObject(ymin, xmin, ymax, xmax, conf_max, chosen_cls));
                }
            }
        }
    }
}

void YoloPost::iou_over_frame() {
    std::sort(detections.begin(), detections.end());
    for (size_t i = 0; i < detections.size(); ++i) {
        if (detections[i].confidence <= conf_threshold) {
            continue;
        }
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if ((detections[i].class_id == detections[j].class_id) && (detections[j].confidence >= conf_threshold)) {
                if (iou(detections[i], detections[j]) >= iou_threshold) {
                    detections[j].confidence = -1.f;
                    num_detections--;
                }
            }
        }
    }
}

const std::vector<DetectionObject> YoloPost::get_detections() const {
    return detections;
}

std::vector<DetectionObject> YoloPost::decode() {
    for (int i = 0; i < feature_maps.size(); ++i) {
        feature_maps[i].extract_boxes(detections, max_num_detections);
    }
    iou_over_frame();
    return detections;
}
