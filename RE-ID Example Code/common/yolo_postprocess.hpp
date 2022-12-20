/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#ifndef _HAILO_YOLO_POST_HPP_
#define _HAILO_YOLO_POST_HPP_
#include "hailo_objects.hpp"
#include "hailo_common.hpp"
#include "yolo_output.hpp"

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
    YoloParams() : iou_threshold(0.45f), detection_threshold(0.35f), output_activation("none"), label_offset(1) {}
    void check_params_logic(uint num_classes_tensors);
};
YoloParams *init(std::string config_path);
void free_resources(void *params_void_ptr);
void filter(HailoROIPtr roi, void *params_void_ptr);
void yolov5(HailoROIPtr roi, void *params_void_ptr);

void yolox(HailoROIPtr roi, void *params_void_ptr);
void yoloxx(HailoROIPtr roi, void *params_void_ptr);
void yolov3(HailoROIPtr roi, void *params_void_ptr);
void yolov4(HailoROIPtr roi, void *params_void_ptr);
void tiny_yolov4_license_plates(HailoROIPtr roi, void *params_void_ptr);
void yolov5_no_persons(HailoROIPtr roi, void *params_void_ptr);
void yolov5_counter(HailoROIPtr roi, void *params_void_ptr);
void yolov5_vehicles_only(HailoROIPtr roi, void *params_void_ptr);
void yolov5_personface(HailoROIPtr roi, void *params_void_ptr);
void yolov5_adas(HailoROIPtr roi, void *params_void_ptr);

__END_DECLS
#endif