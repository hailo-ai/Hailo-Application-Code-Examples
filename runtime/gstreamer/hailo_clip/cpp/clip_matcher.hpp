/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/

#pragma once
#include "hailo_objects.hpp"
#include "hailo_common.hpp"

__BEGIN_DECLS
    void *init(std::string config_path, std::string func_name);
    void filter(HailoROIPtr roi);
    void run(HailoROIPtr roi);
    void update_config(std::string config_path);
__END_DECLS
