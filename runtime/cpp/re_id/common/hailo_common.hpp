/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
/* __BEGIN_DECLS should be used at the beginning of your declarations,
   so that C++ compilers don't mangle their names.  Use __END_DECLS at
   the end of C declarations. */

#pragma once
#include "hailo_objects.hpp"
// #include <stdlib.h>
// #include <string>
// #include <cstring>

#undef __BEGIN_DECLS
#undef __END_DECLS
#ifdef __cplusplus
#define __BEGIN_DECLS \
    extern "C"        \
    {
#define __END_DECLS }
#else
#define __BEGIN_DECLS /* empty */
#define __END_DECLS   /* empty */
#endif

namespace hailo_common
{

    inline void add_object(HailoROIPtr roi, HailoObjectPtr obj)
    {
        roi->add_object(obj);
    }

    inline void add_objects(HailoROIPtr roi, std::vector<HailoObjectPtr> objects)
    {
        for (HailoObjectPtr obj : objects)
        {
            add_object(roi, obj);
        }
    }

    inline void add_classification(HailoROIPtr roi, std::string type, std::string label, float confidence, int class_id = NULL_CLASS_ID)
    {
        add_object(roi,
                   std::make_shared<HailoClassification>(type, class_id, label, confidence));
    }

    inline HailoDetectionPtr add_detection(HailoROIPtr roi, HailoBBox bbox, std::string label, float confidence, int class_id = NULL_CLASS_ID)
    {
        HailoDetectionPtr detection = std::make_shared<HailoDetection>(bbox, class_id, label, confidence);
        detection->set_scaling_bbox(roi->get_bbox());
        add_object(roi, detection);
        return detection;
    }

    inline void add_detections(HailoROIPtr roi, std::vector<HailoDetection> detections)
    {
        for (auto det : detections)
        {
            add_object(roi, std::make_shared<HailoDetection>(det));
        }
    }

    inline void add_detection_pointers(HailoROIPtr roi, std::vector<HailoDetectionPtr> detections)
    {
        for (auto det : detections)
        {
            det->set_scaling_bbox(roi->get_bbox());
            roi->add_object(det);
        }
    }

    inline void remove_objects(HailoROIPtr roi, std::vector<HailoObjectPtr> objects)
    {
        for (HailoObjectPtr obj : objects)
        {
            roi->remove_object(obj);
        }
    }

    inline void remove_detections(HailoROIPtr roi, std::vector<HailoDetectionPtr> objects)
    {
        for (HailoObjectPtr obj : objects)
        {
            roi->remove_object(obj);
        }
    }

    /**
     * Flatten HailoBBox with parent HailoBBox.
     * re scales each bbox values (x,y,width,height min/max values) to match the parent scale.
     *
     * @param[in] bbox    HailoBBox, target bbox to flatten.
     * @param[in] parent_bbox  HailoBBox, parent bbox - to scale to.
     * @return void.
     */
    inline HailoBBox create_flattened_bbox(const HailoBBox &bbox, const HailoBBox &parent_bbox)
    {
        float xmin = parent_bbox.xmin() + bbox.xmin() * parent_bbox.width();
        float ymin = parent_bbox.ymin() + bbox.ymin() * parent_bbox.height();

        float width = bbox.width() * parent_bbox.width();
        float height = bbox.height() * parent_bbox.height();

        return HailoBBox(xmin, ymin, width, height);
    }

    /**
     * Flatten HailoROI subobjects and move them under parent HailoROI.
     * re scales each suboject bbox values to match the parent and move the under the ownership of the parent.
     *
     * @param[in] roi    HailoROIPtr, target roi to flatten.
     * @param[in] parent_roi  HailoROIPtr, parent roi - will take ownership of the subobjects of the target.
     * @param[in] filter_type    hailo_object_t, specific subobject type to flatten.
     * @return void.
     */
    inline void flatten_hailo_roi(HailoROIPtr roi, HailoROIPtr parent_roi, hailo_object_t filter_type)
    {
        std::vector<HailoObjectPtr> objects = roi->get_objects();
        for (uint index = 0; index < objects.size(); index++)
        {
            if (objects[index]->get_type() == filter_type)
            {
                HailoROIPtr sub_obj_roi = std::dynamic_pointer_cast<HailoROI>(objects[index]);
                sub_obj_roi->set_bbox(std::move(create_flattened_bbox(sub_obj_roi->get_bbox(), roi->get_bbox())));
                parent_roi->add_object(sub_obj_roi);
                roi->remove_object(index);
                objects.erase(objects.begin() + index);
                index--;
            }
        }
    }

    inline std::vector<HailoDetectionPtr> get_hailo_detections(HailoROIPtr roi)
    {
        std::vector<HailoObjectPtr> objects = roi->get_objects_typed(HAILO_DETECTION);
        std::vector<HailoDetectionPtr> detections;

        for (auto obj : objects)
        {
            detections.emplace_back(std::dynamic_pointer_cast<HailoDetection>(obj));
        }
        return detections;
    }

    inline std::vector<HailoTileROIPtr> get_hailo_tiles(HailoROIPtr roi)
    {
        std::vector<HailoObjectPtr> objects = roi->get_objects_typed(HAILO_TILE);
        std::vector<HailoTileROIPtr> tiles;

        for (auto obj : objects)
        {
            tiles.emplace_back(std::dynamic_pointer_cast<HailoTileROI>(obj));
        }
        return tiles;
    }

    inline std::vector<HailoClassificationPtr> get_hailo_classifications(HailoROIPtr roi)
    {
        std::vector<HailoObjectPtr> objects = roi->get_objects_typed(HAILO_CLASSIFICATION);
        std::vector<HailoClassificationPtr> classifications;

        for (auto obj : objects)
        {
            classifications.emplace_back(std::dynamic_pointer_cast<HailoClassification>(obj));
        }
        return classifications;
    }

    inline std::vector<HailoUniqueIDPtr> get_hailo_unique_id(HailoROIPtr roi)
    {
        std::vector<HailoObjectPtr> objects = roi->get_objects_typed(HAILO_UNIQUE_ID);
        std::vector<HailoUniqueIDPtr> unique_ids;

        for (auto obj : objects)
        {
            unique_ids.emplace_back(std::dynamic_pointer_cast<HailoUniqueID>(obj));
        }
        return unique_ids;
    }

    inline std::vector<HailoROIPtr> get_hailo_roi_instances(HailoROIPtr roi)
    {
        std::vector<HailoROIPtr> hailo_rois;

        for (auto obj : roi->get_objects())
        {
            HailoROIPtr hailo_roi = std::dynamic_pointer_cast<HailoROI>(obj);
            if (hailo_roi)
                hailo_rois.emplace_back(hailo_roi);
        }
        return hailo_rois;
    }

}