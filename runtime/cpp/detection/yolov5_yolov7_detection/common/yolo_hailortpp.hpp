/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#pragma once
#include "hailo_objects.hpp"
#include "hailo_common.hpp"


__BEGIN_DECLS
void filter_nms(HailoROIPtr roi);
void yolov5_nms(HailoROIPtr roi);
void yolox(HailoROIPtr roi);
void yolov5_no_persons(HailoROIPtr roi);
void yolov5m_vehicles(HailoROIPtr roi);
__END_DECLS