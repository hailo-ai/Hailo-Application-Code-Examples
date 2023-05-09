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
 * @file yolo_post_processing.cpp
 * @brief Yolo Post-Processing
 **/

#include "yolo_post_processing.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <list>
#include <stdint.h>
#include <map>


#define FEATURE_MAP_SIZE1    20
#define FEATURE_MAP_SIZE2    40
#define FEATURE_MAP_SIZE3    80

#define FEATURE_MAP_CHANNELS 85
#define ANCHORS_NUM          3
#define IOU_THRESHOLD        0.6f
#define YOLOV5M_IMAGE_SIZE   640

#define CONF_CHANNEL_OFFSET  4
#define CLASS_CHANNEL_OFFSET 5
#define YMIN                 0
#define XMIN                 1
#define YMAX                 2
#define XMAX                 3
#define CONFIDENCE           4
#define CLASS_ID             5 


float fix_scale(uint8_t& input, float &qp_scale, float &qp_zp)
{
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


void extract_boxes(uint8_t* fm, float &qp_zp, float &qp_scale, int feature_map_size,
		           int* anchors, std::vector<DetectionObject>& objects, float& thr, int max_num_detections) 
{
    float  confidence, x, y, h, w, xmin, ymin, xmax, ymax, conf_max = 0.0f;
    int add = 0, anchor = 0, chosen_row = 0, chosen_col = 0, chosen_cls = -1;
    uint8_t cls_prob, prob_max;
    // channels 0-3 are box coordinates, channel 4 is the confidence, and channels 5-84 are classes

    for (int row = 0; row < feature_map_size; ++row) {
        for (int col = 0; col < feature_map_size; ++col) {
            prob_max = 0;
            for (int a = 0; a < ANCHORS_NUM; ++a) {
                add = FEATURE_MAP_CHANNELS * ANCHORS_NUM * feature_map_size * row + FEATURE_MAP_CHANNELS * ANCHORS_NUM * col + FEATURE_MAP_CHANNELS * a + CONF_CHANNEL_OFFSET;
                confidence = fix_scale(fm[add], qp_scale,  qp_zp);
                if (confidence < thr)
                    continue;
                for (int c = CLASS_CHANNEL_OFFSET; c < FEATURE_MAP_CHANNELS; ++c) {
                    add = FEATURE_MAP_CHANNELS * ANCHORS_NUM * feature_map_size * row + FEATURE_MAP_CHANNELS * ANCHORS_NUM * col + FEATURE_MAP_CHANNELS * a + c;
                    // final confidence: box confidence * class probability
                    cls_prob = fm[add];
                    if (cls_prob > prob_max) 
					{
                        conf_max = fix_scale(cls_prob, qp_scale,  qp_zp) * confidence;
                        chosen_cls = c - CLASS_CHANNEL_OFFSET + 1;
                        prob_max = cls_prob;
                        anchor = a;
                        chosen_row = row;
                        chosen_col = col;
                    }
                }
                if (conf_max >= thr) {
                    add = FEATURE_MAP_CHANNELS * ANCHORS_NUM * feature_map_size * chosen_row + FEATURE_MAP_CHANNELS * ANCHORS_NUM * chosen_col + FEATURE_MAP_CHANNELS * anchor;
                    // box centers
                    x = (fix_scale(fm[add], qp_scale,  qp_zp) * 2.0f - 0.5f + (float)(chosen_col)) / ((float)(feature_map_size));
                    y = (fix_scale(fm[add + 1], qp_scale,  qp_zp) * 2.0f - 0.5f +  (float)(chosen_row)) / (float)(feature_map_size);
                    // box scales
                    w = (float)pow(2.0f * (fix_scale(fm[add + 2], qp_scale,  qp_zp)), 2.0f) * (float)(anchors[anchor * 2]) / YOLOV5M_IMAGE_SIZE;
                    h = (float)pow(2.0f * (fix_scale(fm[add + 3], qp_scale,  qp_zp)), 2.0f) * (float)(anchors[anchor * 2 + 1]) / YOLOV5M_IMAGE_SIZE;
                    // x,y,h,w to xmin,ymin,xmax,ymax
                    xmin = std::max(((x - (w / 2.0f)) * YOLOV5M_IMAGE_SIZE), 0.0f);
                    ymin = std::max(((y - (h / 2.0f)) * YOLOV5M_IMAGE_SIZE), 0.0f);
                    xmax = std::min(((x + (w / 2.0f)) * YOLOV5M_IMAGE_SIZE), (static_cast<float>(YOLOV5M_IMAGE_SIZE) - 1));
                    ymax = std::min(((y + (h / 2.0f)) * YOLOV5M_IMAGE_SIZE), (static_cast<float>(YOLOV5M_IMAGE_SIZE) - 1));

                    if (objects.size() < max_num_detections)
					{
                        objects.push_back(DetectionObject(ymin, xmin, ymax, xmax, conf_max, chosen_cls));
					}
                }
            }
        }
    }
}


std::vector<DetectionObject> _decode(uint8_t* fm1, uint8_t* fm2, uint8_t* fm3, int* anchors1, int* anchors2, int* anchors3,
    float& qp_zp_1, float& qp_scale_1, float& qp_zp_2, float& qp_scale_2, float& qp_zp_3, float& qp_scale_3, float& thr, int max_num_detections)
{
    size_t num_boxes = 0;
    std::vector<DetectionObject> objects;
    objects.reserve(max_num_detections);

    // feature map1
    extract_boxes(fm1, qp_zp_1, qp_scale_1, FEATURE_MAP_SIZE1, anchors1, objects, thr, max_num_detections);

    // feature map2
    extract_boxes(fm2,  qp_zp_2, qp_scale_2, FEATURE_MAP_SIZE2, anchors2, objects, thr, max_num_detections);

    // feature map3
    extract_boxes(fm3,  qp_zp_3, qp_scale_3, FEATURE_MAP_SIZE3, anchors3, objects, thr, max_num_detections);

    num_boxes = objects.size();

    // filter by overlapping boxes
    if (objects.size() > 0) {
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

std::vector<DetectionObject> post_processing(
    int max_num_detections, float thr, std::string &arch,
    uint8_t *fm1, float qp_zp_1, float qp_scale_1,
    uint8_t *fm2, float qp_zp_2, float qp_scale_2,
    uint8_t *fm3, float qp_zp_3, float qp_scale_3)
{

    std::map<std::string, std::vector<std::vector<int>>> arch_to_anchors = {
        {"yolov5", {{116, 90, 156, 198, 373, 326}, {30, 61, 62, 45, 59, 119}, {10, 13, 16, 30, 33, 23}}},
        {"yolov7", {{142, 110, 192, 243, 459, 401}, {36, 75, 76, 55, 72, 146}, {12, 16, 19, 36, 40, 28}}}
    };

    auto anchors = arch_to_anchors[arch];

    return _decode(fm1, fm2, fm3, &anchors[0][0], &anchors[1][0], &anchors[2][0], qp_zp_1, qp_scale_1,
        qp_zp_2, qp_scale_2, qp_zp_3, qp_scale_3, thr, max_num_detections);
    
}
