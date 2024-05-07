/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#include <iostream>
#include "hailo_objects.hpp"
#include "hailo_common.hpp"
#include "aspect_ratio_fix.hpp"

double map(double x, double in_min, double in_max, double out_min, double out_max) {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}


void aspect_ratio_fix(HailoROIPtr roi, double aspect_ratio=16.0/9.0, bool square_bbox=false){
    // scale the bbox to fit original aspect ratio
    // the original aspect ratio is 16:9 for example
    // the inferred image aspect ratio is 1:1 with borders on the top and bottom
    //|----------------------|
    //|    (black border     |
    //|                      |
    //|------top_border------|
    //|                      |
    //|     scaled image     |
    //|                      |
    //|----bottom_border-----|
    //|                      |
    //|    (black border     |
    //|----------------------|   
    double bottom_border = (1.0-1.0/aspect_ratio)/2.0;
    double top_border = 1.0 - bottom_border;
    std::vector<HailoDetectionPtr> detections_ptrs;
    detections_ptrs = hailo_common::get_hailo_detections(roi);
    for (HailoDetectionPtr &detection : detections_ptrs)
    {
        auto bbox = detection->get_bbox();
        // lets map y coordinates to the original image
        double ymin = map(bbox.ymin(), bottom_border, top_border, 0.0, 1.0);
        double ymax = map(bbox.ymax(), bottom_border, top_border, 0.0, 1.0);
        double height = ymax - ymin;
        // get required x coordinates
        double xmin = bbox.xmin();
        double width = bbox.width();
        
        if (square_bbox){
            // in addition we want to get square bboxes to prevent distorsion in the cropper
            // lets get make the bbox square (need to take aspect ratio into account)
            double normalized_height = height / aspect_ratio;
            if (normalized_height > width){
                xmin = xmin + (width - height / aspect_ratio)/2.0;
                width = height / aspect_ratio;
            }
            else if (normalized_height < width){
                ymin = ymin + (height - width * aspect_ratio)/2.0;
                height = width * aspect_ratio;
            }
        }
        HailoBBox new_bbox(xmin, ymin, width, height);
        detection->set_bbox(new_bbox);
    }
}


void filter(HailoROIPtr roi)
{
    // This function is augmenting the detections to fit the original aspect ratio of the image.
    HailoBBox roi_bbox = hailo_common::create_flattened_bbox(roi->get_bbox(), roi->get_scaling_bbox());
    std::vector<HailoDetectionPtr> detections = hailo_common::get_hailo_detections(roi);

    for (auto &detection : detections)
    {
        auto detection_bbox = detection->get_bbox();
        auto xmin = (detection_bbox.xmin() * roi_bbox.width()) + roi_bbox.xmin();
        auto ymin = (detection_bbox.ymin() * roi_bbox.height()) + roi_bbox.ymin();
        auto xmax = (detection_bbox.xmax() * roi_bbox.width()) + roi_bbox.xmin();
        auto ymax = (detection_bbox.ymax() * roi_bbox.height()) + roi_bbox.ymin();

        HailoBBox new_bbox(xmin, ymin, xmax - xmin, ymax - ymin);
        detection->set_bbox(new_bbox);
    }

    // Clear the scaling bbox of main roi because all detections are fixed.
    roi->clear_scaling_bbox();
}

// To fix 16:9 aspect ratio
void fix_16_9(HailoROIPtr roi)
{
    aspect_ratio_fix(roi, 16.0/9.0);
}

// To fix 4:3 aspect ratio
void fix_4_3(HailoROIPtr roi)
{
    aspect_ratio_fix(roi, 4.0/3.0);
}

// To fix 16:9 aspect ratio with square bboxes
void fix_16_9_square(HailoROIPtr roi)
{
    aspect_ratio_fix(roi, 16.0/9.0, true);
}

// To fix 4:3 aspect ratio with square bboxes
void fix_4_3_spuare(HailoROIPtr roi)
{
    aspect_ratio_fix(roi, 4.0/3.0, true);
}

