/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#ifndef _HAILO_YOLACT_TWENTY_HPP_
#define _HAILO_YOLACT_TWENTY_HPP_
#include <map>
namespace common
{
    static std::map<uint8_t, std::string> yolact_twenty = {
        {0, "background"},
        {1, "person"},
        {2, "bicycle"},
        {3, "bench"},
        {4, "backpack"},
        {5, "handbag"},
        {6, "tie"},
        {7, "suitcase"},
        {8, "bottle"},
        {9, "cup"},
        {10, "bowl"},
        {11, "chair"},
        {12, "couch"},
        {13, "potted plant"},
        {14, "dining table"},
        {15, "tv"},
        {16, "laptop"},
        {17, "keyboard"},
        {18, "cellphone"},
        {19, "book"}};
}
#endif /* _HAILO_YOLACT_TWENTY_HPP_ */