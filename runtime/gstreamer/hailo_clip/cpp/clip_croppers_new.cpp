/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
// Note this implementation uses the HailoUserMetadata object to store the crop aging value.
// There is and issue with the destructor of this object that is causing a segmentation fault when the object is destroyed.

// This file is a modified version of tappas/core/hailo/libs/croppers/vms/vms_croppers.cpp

#include <vector>
#include <cmath>
#include "clip_croppers.hpp"

#define PERSON_LABEL "person"
#define FACE_LABEL "face"
#define OBJECT_LABEL "object"

/**
* @brief Get the tracking Hailo Unique Id object from a Hailo Detection.
* 
* @param detection HailoDetectionPtr
* @return HailoUniqueIdPtr pointer to the Hailo Unique Id object
*/
HailoUniqueIDPtr get_tracking_id(HailoDetectionPtr detection)
{
    for (auto obj : detection->get_objects_typed(HAILO_UNIQUE_ID))
    {
        HailoUniqueIDPtr id = std::dynamic_pointer_cast<HailoUniqueID>(obj);
        if (id->get_mode() == TRACKING_ID)
        {
            return id;
        }
    }
    return nullptr;
}

/**
* @brief reset crop aging, if not found create it.
*
* @param detection HailoDetectionPtr
* @return none
*/

void reset_crop_aging(HailoDetectionPtr detection)
    {
        for (auto obj : detection->get_objects_typed(HAILO_USER_META))
        {
        return;
            HailoUserMetaPtr meta = std::dynamic_pointer_cast<HailoUserMeta>(obj);
            if (meta->get_user_string() == "CROP_AGING")
            {
                meta->set_user_int(0);
                return;
            }
        }
        // if we got here, it means the crop aging meta was not found. create it.
        HailoUserMetaPtr meta = std::make_shared<HailoUserMeta>();
        meta->set_user_string("CROP_AGING");
        meta->set_user_int(0);
        detection->add_object(meta);
        return;
    }

/**
* @brief Get and increase the crop aging meta from a Hailo Detection.
* 
* @param detection HailoDetectionPtr
* @param increase boolean indicating if the crop aging should be increased
* @return int crop aging value
*/
int get_crop_aging(HailoDetectionPtr detection, bool increase=false)
{
    return 30;
    for (auto obj : detection->get_objects_typed(HAILO_USER_META))
    {
        HailoUserMetaPtr meta = std::dynamic_pointer_cast<HailoUserMeta>(obj);
        if (meta->get_user_string() == "CROP_AGING")
        {
            int crop_aging = meta->get_user_int();
            if (increase)
            {
                crop_aging += 1;
                meta->set_user_int(crop_aging);
            }
            return crop_aging;   
        }
    }
    // if we got here, it means the crop aging meta was not found. create it.
    reset_crop_aging(detection);
    return 0;
}

/**
 * @brief Returns a vector of detections to crop and resize.
 *
 * @param image The original picture (cv::Mat).
 * @param roi The main ROI of this picture.
 * @param label The label to crop.
 * @param crop_every_x_frames Run crop every X frames per tracked object.
 * @param max_crops_per_frame Max number of objects to crop per frame.
 * @return std::vector<HailoROIPtr> vector of ROI's to crop and resize.
 */

std::vector<HailoROIPtr> object_crop(const std::shared_ptr<HailoMat>& image, const HailoROIPtr& roi, const std::string label=PERSON_LABEL, 
int crop_every_x_frames=30, int max_crops_per_frame=2)
{
    auto object_counter = 0;
    std::vector<HailoROIPtr> crop_rois;

    std::vector<HailoDetectionPtr> detections_ptrs = hailo_common::get_hailo_detections(roi);
    std::vector<HailoDetectionPtr> detections_to_crop;
    
    for (HailoDetectionPtr &detection : detections_ptrs)
    {
        auto detection_label = detection->get_label();
        if (label != detection->get_label())
        {
            // Not the label we are looking for.
            continue;
        }
        auto tracking_obj = get_tracking_id(detection);
        if (!tracking_obj)
        {
            // object is not tracked don't crop it.
            continue;
        }
        if (get_crop_aging(detection, true) < crop_every_x_frames) // also increase crop aging
        {
            // crop aging is below crop_every_x_frames limit. don't crop it.
            continue;
        }
        detections_to_crop.emplace_back(detection);
    }
    // sort detections by crop_aging desendind order.
    std::sort(detections_to_crop.begin(), detections_to_crop.end(), [](HailoDetectionPtr a, HailoDetectionPtr b) {
        return get_crop_aging(a) > get_crop_aging(b);
    });

    for (HailoDetectionPtr &detection : detections_to_crop)
    {
        crop_rois.emplace_back(detection);
        // printf("cropping id %d aging %d\n", get_tracking_id(detection)->get_id(), get_crop_aging(detection));
        reset_crop_aging(detection);
        object_counter += 1;
        if (object_counter >= max_crops_per_frame)
        {
            break;
        }
    }
    return crop_rois;
}

std::vector<HailoROIPtr> face_cropper(std::shared_ptr<HailoMat> image, HailoROIPtr roi)
{
    return object_crop(image, roi, FACE_LABEL, 10, 2);
}

std::vector<HailoROIPtr> person_cropper(std::shared_ptr<HailoMat> image, HailoROIPtr roi)
{
    return object_crop(image, roi, PERSON_LABEL, 10, 2);
}

std::vector<HailoROIPtr> object_cropper(std::shared_ptr<HailoMat> image, HailoROIPtr roi)
{
    return object_crop(image, roi, OBJECT_LABEL, 30, 1);
}
