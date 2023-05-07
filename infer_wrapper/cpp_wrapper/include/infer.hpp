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

// class TensorWrapper {
//     public:
//     float array_5[5]; // = new float[5];
//     float array_6[6]; // = new float[6];
//     float array_7[7]; // = new float[7];
// };

extern "C" int infer_wrapper(const char* hef_path, const char* images_path, 
    float32_t* arr1, size_t n1,
    float32_t* arr2, size_t n2,
    float32_t* arr3, size_t n3);

#endif /* _HAILO_INFER_HPP_ */