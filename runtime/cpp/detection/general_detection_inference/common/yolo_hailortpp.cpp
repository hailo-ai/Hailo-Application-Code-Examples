#include "hailo_nms_decode.hpp"
#include "yolo_hailortpp.hpp"
#include "labels/coco_eighty.hpp"
#include "labels/coco_ninety.hpp"

#include <regex>


void filter(HailoROIPtr roi, void *params_void_ptr) {
    if (!roi->has_tensors())
    {
        return;
    }

    std::vector<HailoTensorPtr> tensors = roi->get_tensors();
    std::map<uint8_t, std::string> labels_map;

    if (tensors[0]->name().find("mobilenet") != std::string::npos)
        labels_map = common::coco_ninety_classes;
    else
        labels_map = common::coco_eighty;

    // find the nms tensor
    for (auto tensor : tensors)
    {
        if (std::regex_search(tensor->name(), std::regex("nms"))) 
        {
            auto post = HailoNMSDecode(tensor, labels_map);
            auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
            hailo_common::add_detections(roi, detections);
        }
    }
}
