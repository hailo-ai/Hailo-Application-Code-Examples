/**
 * Copyright (c) 2020-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
/**
 * @file infer.hpp
 * @brief Inference
 **/

#ifndef _HAILO_INFER_HPP_
#define _HAILO_INFER_HPP_

extern "C" int infer_wrapper(const char* hef_path, const char* image_path, const char* arch, float* detections, int max_num_detections);

#endif /* _HAILO_INFER_HPP_ */