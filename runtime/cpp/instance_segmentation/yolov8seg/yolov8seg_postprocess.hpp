/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#pragma once
#include "common/hailo_objects.hpp"
#include "common/hailo_common.hpp"

#include <opencv2/opencv.hpp>

#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor-blas/xlinalg.hpp>

struct Quadruple {
    std::vector<HailoTensorPtr> boxes;
    xt::xarray<float> scores;
    std::vector<HailoTensorPtr> masks;
    xt::xarray<float> proto_data;
};

struct DetectionAndMask {
    HailoDetection detection;
    cv::Mat mask;
};

__BEGIN_DECLS
std::vector<cv::Mat> filter(HailoROIPtr roi, int org_width, int org_height);
__END_DECLS
