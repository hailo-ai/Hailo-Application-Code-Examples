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

extern "C" int infer_wrapper(const char* hef_path, const char* images_path, const char* arch, const int conf_thr, float* detections, const int max_num_detections, int* frames_ready, const int buffer_size);

#endif /* _HAILO_INFER_HPP_ */