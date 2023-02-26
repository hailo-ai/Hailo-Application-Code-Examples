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
 * @file ssd_post_processing.cpp
 * @brief SSD Post-Processing
 **/

#include "ssd_post_processing.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdint.h>


float fix_scale(uint8_t& input, float &qp_scale, float &qp_zp)
{
    /* Quantized to Native*/
    return (float(input) - qp_zp) * qp_scale;
}

float iou_calc(const DetectionObject &box_1, const DetectionObject &box_2)
{
    const float width_of_overlap_area = std::min(box_1.xmax, box_2.xmax) - std::max(box_1.xmin, box_2.xmin);
    const float height_of_overlap_area = std::min(box_1.ymax, box_2.ymax) - std::max(box_1.ymin, box_2.ymin);
    const float positive_width_of_overlap_area = std::max(width_of_overlap_area, 0.0f);
    const float positive_height_of_overlap_area = std::max(height_of_overlap_area, 0.0f);
    const float area_of_overlap = positive_width_of_overlap_area * positive_height_of_overlap_area;
    const float box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin);
    const float box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin);
    return area_of_overlap / (box_1_area + box_2_area - area_of_overlap);
}


static float sigmoid(const float x)
{
    /* returns the value of the sigmoid function f(x) = 1/(1 + e^-x) */
    return 1.0f / (1.0f + expf(-x));
}


void ssd_extract_boxes(std::pair<OutTensor,OutTensor> &tensors,
		           std::vector<float> branch_anchors, std::vector<DetectionObject>& objects, float& thr)
{
    // OutTensor reg_tensor = tensors.first; // in default ssd, 12 or 24
    // OutTensor cls_tensor = tensors.second; // in default ssd, 273 or 546
    int feature_map_width = tensors.first.width;
    int feature_map_height = tensors.first.height;
    int num_anchors = int(branch_anchors.size()/2);
    int num_classes = int(tensors.second.channels / num_anchors);

    for (int row = 0; row < feature_map_height; ++row) {
        for (int col = 0; col < feature_map_width; ++col) {
            for (int idx_anchor = 0; idx_anchor < num_anchors; ++idx_anchor) {
                std::pair<uint32_t, float32_t> max_id_score_pair = {0, -1.f};
                for (int idx_class = 1; idx_class < num_classes; ++idx_class){ // starting without background class.
                    // access index for class
                    uint32_t access_cls = (col * num_anchors * num_classes) + (row * feature_map_height * num_anchors * num_classes) + (idx_anchor * num_classes) + idx_class;
                    auto class_confidence = fix_scale(tensors.second.m_data[access_cls], tensors.second.qp_scale, tensors.second.qp_zp);
                    class_confidence = sigmoid(class_confidence);
                    if (class_confidence > max_id_score_pair.second) { 
                        max_id_score_pair.first = idx_class;
                        max_id_score_pair.second = class_confidence;
                    }
                }

                // access index for bbox
                uint32_t access_bbox = (col * num_anchors * 4) + (row * feature_map_height * num_anchors * 4) + (idx_anchor * 4);

                if (max_id_score_pair.second >= thr) {
                    const auto &ha = branch_anchors[idx_anchor * 2];
                    const auto &wa = branch_anchors[idx_anchor * 2 + 1];

                    const auto xcenter_a = (static_cast<float32_t>(col) + 0.5f) / static_cast<float32_t>(tensors.first.width);
                    const auto ycenter_a = (static_cast<float32_t>(row) + 0.5f) / static_cast<float32_t>(tensors.first.height);

                    auto ty = fix_scale(tensors.first.m_data[access_bbox], tensors.first.qp_scale, tensors.first.qp_zp); 
                    auto tx = fix_scale(tensors.first.m_data[access_bbox+1], tensors.first.qp_scale, tensors.first.qp_zp);
                    auto th = fix_scale(tensors.first.m_data[access_bbox+2], tensors.first.qp_scale, tensors.first.qp_zp);
                    auto tw = fix_scale(tensors.first.m_data[access_bbox+3], tensors.first.qp_scale, tensors.first.qp_zp);

                    // scale factor
                    ty /= BOX_CODER_SCALE[0];
                    tx /= BOX_CODER_SCALE[1];
                    th /= BOX_CODER_SCALE[2];
                    tw /= BOX_CODER_SCALE[3];

                    float w = static_cast<float32_t>(exp(tw)) * wa;
                    float h = static_cast<float32_t>(exp(th)) * ha;
                    auto x_center = tx * wa + xcenter_a;
                    auto y_center = ty * ha + ycenter_a;

                    auto x_min = std::max((x_center - (w / 2.0f)) , 0.0f); 
                    auto y_min = std::max((y_center - (h / 2.0f)) , 0.0f);
                    auto x_max = std::min((x_center + (w / 2.0f)), 1.f);
                    auto y_max = std::min((y_center + (h / 2.0f)), 1.f);

                    if (objects.size() < MAX_BOXES){
                        objects.push_back(DetectionObject(y_min, x_min, y_max, x_max, max_id_score_pair.second, max_id_score_pair.first));
                    }
                    else return;
                }
            }
        }
    }
}


std::vector<DetectionObject> ssd_decode(std::vector<std::pair<OutTensor,OutTensor>> &tensors, std::vector<std::vector<float>> &anchors, float& thr)
{
    std::vector<DetectionObject> objects;
    objects.reserve(MAX_BOXES);
    if (tensors.size() <= 0) return objects;

    for(size_t i = 0; i < tensors.size(); i++){
        ssd_extract_boxes(tensors[i], anchors[i], objects, thr);
    }
    size_t num_boxes = objects.size();

    // filter by overlapping boxes
    if(objects.size() > 0) {
        std::sort(objects.begin(), objects.end());
        for (unsigned int i = 0; i < objects.size(); ++i) {
            if (objects[i].confidence <= thr) {
                continue;
            }
            for (unsigned int j = i + 1; j < objects.size(); ++j) {
                if ((objects[i].class_id == objects[j].class_id) && (objects[j].confidence >= thr)) {
                    if (iou_calc(objects[i], objects[j]) >= IOU_THRESHOLD) {
                        objects[j].confidence = -1.f;
                        num_boxes -= 1;
                    }
                }
            }
        }
    }
    return objects;
}

std::vector<DetectionObject> post_processing(std::vector<std::pair<OutTensor,OutTensor>> &tensors)
{

    std::vector<std::vector<float>> anchors;
    // h0, w0, ... , hn, wn. values from config json
    std::vector<float> anchor1 = {0.1, 0.1, 0.1414213562373095, 0.282842712474619, 0.282842712474619, 0.1414213562373095};
    std::vector<float> anchor2 = {0.35, 0.35, 0.2474873734152916, 0.4949747468305833, 0.4949747468305832, 0.24748737341529164, 0.20207259421636903, 0.606217782649107, 0.6062480958117455, 0.20206249033405482, 0.4183300132670378, 0.4183300132670378};
    std::vector<float> anchor3 = {0.5, 0.5, 0.35355339059327373, 0.7071067811865476, 0.7071067811865475, 0.3535533905932738, 0.2886751345948129, 0.8660254037844386, 0.8660687083024937, 0.2886607004772212, 0.570087712549569, 0.570087712549569};
    std::vector<float> anchor4 = {0.65, 0.65, 0.4596194077712559, 0.9192388155425119, 0.9192388155425117, 0.45961940777125593, 0.37527767497325676, 1.12583302491977, 1.1258893207932419, 0.3752589106203876, 0.7211102550927979, 0.7211102550927979};
    std::vector<float> anchor5 = {0.8, 0.8, 0.565685424949238, 1.1313708498984762, 1.131370849898476, 0.5656854249492381, 0.46188021535170065, 1.3856406460551018, 1.38570993328399, 0.46185712076355395, 0.8717797887081347, 0.8717797887081347};
    std::vector<float> anchor6 = {0.95, 0.95, 0.67175144212722, 1.3435028842544403, 1.34350288425444, 0.6717514421272202, 0.5484827557301445, 1.6454482671904334, 1.645530545774738, 0.5484553309067203, 0.9746794344808963, 0.9746794344808963};
    anchors.insert(anchors.end(), { anchor1, anchor2, anchor3, anchor4, anchor5, anchor6});

    float thr = CONFIDENCE_THRESHOLD;
    std::vector<DetectionObject> objects = ssd_decode(tensors, anchors, thr);

    return objects;
}

