/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#pragma once
#include "common/hailo_objects.hpp"
#include "common/hailo_common.hpp"
#include "yolo_output.hpp"
#include "common/labels/coco_eighty.hpp"

__BEGIN_DECLS

class YoloParams
{
public:
    float iou_threshold;
    float detection_threshold;
    std::map<std::uint8_t, std::string> labels;
    uint num_classes;
    uint max_boxes;
    std::vector<std::vector<int>> anchors_vec;
    std::string output_activation; // can be "none" or "sigmoid"
    int label_offset;
    YoloParams() : iou_threshold(0.45f), detection_threshold(0.3f), output_activation("none"), label_offset(1) {}
    void check_params_logic(uint num_classes_tensors);
};

class Yolov5Params : public YoloParams
{
public:
    Yolov5Params()
    {
        labels = common::coco_eighty;
        max_boxes = 200;
        anchors_vec = {
            {116, 90, 156, 198, 373, 326},
            {30, 61, 62, 45, 59, 119},
            {10, 13, 16, 30, 33, 23}};
    }
};

class Yolov7Params : public YoloParams
{
public:
    Yolov7Params()
    {
        labels = common::coco_eighty;
        max_boxes = 200;
        anchors_vec = {
            {142, 110, 192, 243, 459, 401}, 
            {36, 75, 76, 55, 72, 146}, 
            {12, 16, 19, 36, 40, 28}
        };
    }
};

YoloParams *init(std::string config_path, std::string func_name);
void free_resources(void *params_void_ptr);
void filter(HailoROIPtr roi, void *params_void_ptr);
void yolov5(HailoROIPtr roi, void *params_void_ptr);

__END_DECLS