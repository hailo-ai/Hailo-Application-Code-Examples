/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#include <vector>
#include "common/tensors.hpp"
#include "common/math.hpp"
#include "hailo_tracker.hpp"
#include "hailo_xtensor.hpp"
#include "hailo_objects.hpp"
#include "hailo_common.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"

#include "clip_matcher.hpp"
#include "TextImageMatcher.hpp"
TextImageMatcher* matcher = TextImageMatcher::getInstance("", 0.8f, 6);

static xt::xarray<float> get_xtensor(HailoMatrixPtr matrix)
{
    // Adapt a HailoTensorPtr to an xarray (quantized)
    xt::xarray<float> xtensor = xt::adapt(matrix->get_data().data(), matrix->size(), xt::no_ownership(), matrix->shape());
    // remove (squeeze) xtensor first dim
    xtensor = xt::squeeze(xtensor, 0);
    return xtensor;
}


void* init(std::string config_path, std::string func_name)
{
    if (config_path == "NULL")
    {
        std::cout << "No default JSON provided" << std::endl;
    }
    else 
    {
        matcher->load_embeddings(config_path);
        matcher->run_softmax = false;
    }
    return nullptr;
}

void update_config(std::string config_path)
{
    if (config_path == "NULL")
    {
        std::cout << "No default JSON provided" << std::endl;
    }
    else 
    {
        matcher->load_embeddings(config_path);
    }
    return;
}

void filter(HailoROIPtr roi)
{
    // define 2D array for image embedding
    xt::xarray<double> image_embedding;
    
    // vector to hold detections
    std::vector<HailoDetectionPtr> detections_ptrs;
    
    // vector to hold used detections
    // std::vector<HailoDetectionPtr> used_detections;
    std::vector<HailoROIPtr> used_detections;
    
    // Check if roi is used for clip
    auto roi_matrixs = roi->get_objects_typed(HAILO_MATRIX);
    if (!roi_matrixs.empty())
    {
        HailoMatrixPtr matrix_ptr = std::dynamic_pointer_cast<HailoMatrix>(roi_matrixs[0]);
        xt::xarray<float> embeddings = get_xtensor(matrix_ptr);
        image_embedding = embeddings;
        used_detections.push_back(roi);
    }
    else
    {
        // Get detections from roi
        detections_ptrs = hailo_common::get_hailo_detections(roi);
        
        for (HailoDetectionPtr &detection : detections_ptrs)
        {
            auto matrix_objs = detection->get_objects_typed(HAILO_MATRIX);
            for (auto matrix : matrix_objs)
            {
                HailoMatrixPtr matrix_ptr = std::dynamic_pointer_cast<HailoMatrix>(matrix);
                xt::xarray<float> embeddings = get_xtensor(matrix_ptr);
                // if image_embedding is empty or 0-dimensional, initialize it with embeddings
                if (image_embedding.size() == 0 || image_embedding.dimension() == 0)
                {
                    image_embedding = embeddings;
                }
                else 
                {
                    // if image_embedding is not empty and not 0-dimensional, concatenate it with embeddings
                    image_embedding = xt::concatenate(xt::xtuple(image_embedding, embeddings), 0);
                }
                used_detections.push_back(detection);
            }
        }
    }
    // if image_embedding is empty, return
    if (image_embedding.size() == 0 || image_embedding.dimension() == 0)
    {
        return;
    }
    std::vector<Match> matches = matcher->match(image_embedding);
    for (auto &match : matches)
    {
        auto detection = used_detections[match.row_idx];
        auto old_classifications = hailo_common::get_hailo_classifications(detection);
        for (auto old_classification : old_classifications)
        {
            if (old_classification->get_classification_type() == "clip")
            detection->remove_object(old_classification);
        }
        if (match.negative || !match.passed_threshold)
        {
            continue;
        }
        HailoClassificationPtr classification = std::make_shared<HailoClassification>(std::string("clip"), match.text, match.similarity);
        detection->add_object(classification);
    }
}

void run(HailoROIPtr roi)
{
    filter(roi);
}
 