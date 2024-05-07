/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/

#pragma once
#include "hailo_objects.hpp"

__BEGIN_DECLS
    void filter(HailoROIPtr roi); // default fix 16/9 aspect ratio
    void fix_16_9(HailoROIPtr roi); // To fix 16:9 aspect ratio
    void fix_4_3(HailoROIPtr roi); // To fix 4:3 aspect ratio    
    void fix_16_9_square(HailoROIPtr roi); // To fix 16:9 aspect ratio with square bboxes
    void fix_4_3_spuare(HailoROIPtr roi); // To fix 4:3 aspect ratio with square bboxes
__END_DECLS
