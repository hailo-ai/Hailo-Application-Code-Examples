/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#pragma once
#include "common/hailo_objects.hpp"
#include "common/hailo_common.hpp"

__BEGIN_DECLS
void nanodet_repvgg(HailoROIPtr roi);
void filter(HailoROIPtr roi);
__END_DECLS
