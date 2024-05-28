#include "hailo_nms_decode.hpp"
#include "yolo_hailortpp.hpp"
#include "labels/coco_eighty.hpp"

void yolov8_nms(HailoROIPtr roi, std::string output_name)
{
    auto post = HailoNMSDecode(roi->get_tensor(output_name), common::coco_eighty);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void filter(HailoROIPtr roi, std::string output_name)
{
    yolov8_nms(roi, output_name);
}
