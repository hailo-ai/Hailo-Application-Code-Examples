/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#include <opencv2/opencv.hpp>
#include "hailo_objects.hpp"

// void filter(HailoROIPtr roi, cv::Mat frame, char *current_stream_id);
void filter(HailoROIPtr roi, cv::Mat& frame);
HailoUniqueIDPtr get_global_id(HailoDetectionPtr detection);
cv::Scalar indexToColor(size_t index);