/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#include <vector>
#include "common/tensors.hpp"
#include "common/math.hpp"
#include "clip.hpp"
#include "hailo_tracker.hpp"
#include "hailo_xtensor.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"

#define OUTPUT_LAYER_NAME "clip_resnet_50x4/conv89"


ClipParams *init(std::string config_path, std::string func_name)
{
    if (config_path == "NULL")
    {
        config_path = "hailo_tracker";
    }
    ClipParams *params = new ClipParams(config_path);
    return params;
}

void clip(HailoROIPtr roi, std::string layer_name, std::string tracker_name)
{
    if (!roi->has_tensors())
    {
        return;
    }
    std::string jde_tracker_name = tracker_name + "_" + roi->get_stream_id();
    // std::cout << "jde_tracker_name: " << jde_tracker_name << std::endl;
    // // Get the list of trackers
    // std::vector<std::string> trackers_list = HailoTracker::GetInstance().get_trackers_list();
    // // print the list of trackers
    // std::cout << "Trackers list: " << std::endl;
    // for (auto &tracker : trackers_list)
    // {
    //     std::cout << tracker << std::endl;
    // }
    auto unique_ids = hailo_common::get_hailo_track_id(roi);
    // Remove previous matrices
    if(unique_ids.empty())
        roi->remove_objects_typed(HAILO_MATRIX);
    else
    {
        HailoTracker::GetInstance().remove_matrices_from_track(jde_tracker_name, unique_ids[0]->get_id());
    }
    // Convert the tensor to xarray.
    auto tensor = roi->get_tensor(layer_name);
    xt::xarray<float> embeddings = common::get_xtensor_float(tensor);

    // vector normalization
    auto normalized_embedding = common::vector_normalization(embeddings);
    HailoMatrixPtr hailo_matrix = hailo_common::create_matrix_ptr(normalized_embedding);
    if(unique_ids.empty())
    {
        roi->add_object(hailo_matrix);
    }
    else
    {
        // Update the tracker with the results
        HailoTracker::GetInstance().add_object_to_track(jde_tracker_name,
                                                        unique_ids[0]->get_id(),
                                                        hailo_matrix);
    }
}

void filter(HailoROIPtr roi, void *params_void_ptr)
{
    ClipParams *params = reinterpret_cast<ClipParams *>(params_void_ptr);
    clip(roi, OUTPUT_LAYER_NAME, params->tracker_name);
}
