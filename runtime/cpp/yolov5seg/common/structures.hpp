/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#pragma once

namespace common
{

    //-------------------------------
    // COMMON STRUCTURES
    //-------------------------------
    typedef struct
    {
        float32_t y_min;
        float32_t x_min;
        float32_t y_max;
        float32_t x_max;
        float32_t score;
    } hailo_bbox_float32_t;

    typedef struct
    {
        uint16_t y_min;
        uint16_t x_min;
        uint16_t y_max;
        uint16_t x_max;
        uint16_t score;
    } hailo_bbox_t;

}