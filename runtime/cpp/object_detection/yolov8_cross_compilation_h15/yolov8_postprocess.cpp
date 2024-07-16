/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
// General includes
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <string>
#include <tuple>
#include <vector>

// Hailo includes
#include "common/math.hpp"
#include "common/hailo_nms_decode.hpp"
#include "common/labels/coco_eighty.hpp"
#include "common/hailo_common.hpp"
#include "yolov8_postprocess.hpp"

/**
 * @brief yolov8 postprocess
 *        Provides network specific paramters
 * 
 * @param roi  -  HailoROIPtr
 *        The roi that contains the ouput tensors
 */
void yolov8(HailoROIPtr roi, std::string output_name)
{
    std::vector<HailoTensorPtr> tensors = roi->get_tensors();
    auto post = HailoNMSDecode(roi->get_tensor(output_name), common::coco_eighty);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

//******************************************************************
//  DEFAULT FILTER
//******************************************************************
void filter(HailoROIPtr roi, std::string output_name)
{
    yolov8(roi, output_name);
}
