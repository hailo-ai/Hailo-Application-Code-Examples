/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#pragma once

#include <map>
namespace common
{
    static std::map<uint8_t, std::string> coco_visdrone_classes = {
        {0,"unlabeled"},
        {1,"person"},
        {2,"person"},
        {3,"bicycle"},
        {4,"car"},
        {5,"van"},
        {6,"truck"},
        {7,"tricycle"},
        {8,"awning-tricycle"},
        {9,"bus"},
        {10,"motor"},
        {11, "others"}
    };
}