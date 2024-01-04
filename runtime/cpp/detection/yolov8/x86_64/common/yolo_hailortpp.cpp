#include "hailo_nms_decode.hpp"
#include "yolo_hailortpp.hpp"
#include "labels/coco_eighty.hpp"

static const std::string DEFAULT_YOLOV8M_OUTPUT_LAYER = "yolov8_nms_postprocess";

void yolov8_nms(HailoROIPtr roi, std::string model_type)
{
    std::string output_name = model_type + "/" + DEFAULT_YOLOV8M_OUTPUT_LAYER;
    auto post = HailoNMSDecode(roi->get_tensor(output_name), common::coco_eighty);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void filter_nms(HailoROIPtr roi, std::string model_type)
{
    yolov8_nms(roi, model_type);
}
