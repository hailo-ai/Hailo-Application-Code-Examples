#pragma once

#include "stage.hpp"
#include "buffer.hpp"
#include "hailo_common.hpp"
#include <algorithm>

class AggregatorStage : public ConnectedStage
{
protected:
    bool m_blocking;
    std::string m_main_inlet_name;
    size_t m_main_queue_size;
    std::string m_sub_inlet_name;
    size_t m_sub_queue_size;
    bool m_multi_scale;
    float m_iou_threshold;
    float m_border_threshold;

public:
    AggregatorStage(std::string name, bool blocking, 
                    std::string main_inlet_name, size_t main_queue_size, 
                    std::string sub_inlet_name, size_t sub_queue_size, 
                    bool multi_scale=false, float iou_threshold=0.3, float m_border_threshold=0.1,
                    bool leaky=false, bool print_fps=false) : 
        ConnectedStage(name, main_queue_size, leaky, print_fps), m_blocking(blocking), 
        m_main_inlet_name(main_inlet_name), m_main_queue_size(main_queue_size), 
        m_sub_inlet_name(sub_inlet_name), m_sub_queue_size(sub_queue_size), 
        m_multi_scale(multi_scale), m_iou_threshold(iou_threshold), m_border_threshold(m_border_threshold)
        {
            m_queues.push_back(std::make_shared<Queue>(m_main_inlet_name, m_main_queue_size, leaky));
            m_queues.push_back(std::make_shared<Queue>(m_sub_inlet_name, m_sub_queue_size, leaky));
        }

    void add_queue(std::string name) override {}

    HailoBBox create_flattened_bbox(const HailoBBox &bbox, const HailoBBox &parent_bbox)
    {
        float xmin = parent_bbox.xmin() + bbox.xmin() * parent_bbox.width();
        float ymin = parent_bbox.ymin() + bbox.ymin() * parent_bbox.height();

        float width = bbox.width() * parent_bbox.width();
        float height = bbox.height() * parent_bbox.height();

        return HailoBBox(xmin, ymin, width, height);
    }

    void flatten_hailo_roi(HailoROIPtr roi, HailoROIPtr parent_roi, hailo_object_t filter_type)
    {
        std::vector<HailoObjectPtr> objects = roi->get_objects();
        for (uint index = 0; index < objects.size(); index++)
        {
            if (objects[index]->get_type() == filter_type)
            {
                HailoROIPtr sub_obj_roi = std::dynamic_pointer_cast<HailoROI>(objects[index]);
                sub_obj_roi->set_bbox(std::move(create_flattened_bbox(sub_obj_roi->get_bbox(), roi->get_scaling_bbox())));
                parent_roi->add_object(sub_obj_roi);
                roi->remove_object(index);
                objects.erase(objects.begin() + index);
                index--;
            }
        }
    }

    /**
     * Remove detections close to the boundary of the tile.
     * Not including tile borders that located on one of the borders of the full frame.
     *
     * @param[in] hailo_tile_roi  HailoTileROIPtr taken from the buffer.
     * @param[in] border_threshold    float.  threshold - 0 - 1 value of 'close to border' ratio.
     * @return void.
     */
    static void remove_exceeded_bboxes(HailoROIPtr hailo_tile_roi, float border_threshold)
    {
        auto detections = hailo_common::get_hailo_detections(hailo_tile_roi);
        HailoBBox tile_bbox = hailo_tile_roi->get_scaling_bbox();

        for (const HailoDetectionPtr &detection : detections)
        {
            HailoBBox bbox = detection->get_bbox();
            bool exceed_xmin = (tile_bbox.xmin() != 0 && bbox.xmin() < border_threshold);
            bool exceed_xmax = (tile_bbox.xmax() != 1 && (1 - bbox.xmax()) < border_threshold);
            bool exceed_ymin = (tile_bbox.ymin() != 0 && bbox.ymin() < border_threshold);
            bool exceed_ymax = (tile_bbox.ymax() != 1 && (1 - bbox.ymax()) < border_threshold);

            if (exceed_xmin || exceed_xmax || exceed_ymin || exceed_ymax)
                hailo_tile_roi->remove_object(detection);
        }
    }

    float iou_calc(const HailoBBox &box_1, const HailoBBox &box_2)
    {
        // Calculate IOU between two detection boxes
        const float width_of_overlap_area = std::min(box_1.xmax(), box_2.xmax()) - std::max(box_1.xmin(), box_2.xmin());
        const float height_of_overlap_area = std::min(box_1.ymax(), box_2.ymax()) - std::max(box_1.ymin(), box_2.ymin());
        const float positive_width_of_overlap_area = std::max(width_of_overlap_area, 0.0f);
        const float positive_height_of_overlap_area = std::max(height_of_overlap_area, 0.0f);
        const float area_of_overlap = positive_width_of_overlap_area * positive_height_of_overlap_area;
        const float box_1_area = (box_1.ymax() - box_1.ymin()) * (box_1.xmax() - box_1.xmin());
        const float box_2_area = (box_2.ymax() - box_2.ymin()) * (box_2.xmax() - box_2.xmin());
        // The IOU is a ratio of how much the boxes overlap vs their size outside the overlap.
        // Boxes that are similar will have a higher overlap threshold.
        return area_of_overlap / (box_1_area + box_2_area - area_of_overlap);
    }

    /**
     * @brief Perform IOU based NMS on detection objects of HailoRoi
     *
     * @param hailo_roi  -  HailoROIPtr
     *        The HailoROI contains detections to perform NMS on.
     *
     * @param iou_thr  -  float
     *        Threshold for IOU filtration
     */
    void nms(HailoROIPtr hailo_roi, const float iou_thr)
    {
        // The network may propose multiple detections of similar size/score,
        // which are actually the same detection. We want to filter out the lesser
        // detections with a simple nms.

        std::vector<HailoDetectionPtr> objects = hailo_common::get_hailo_detections(hailo_roi);
        std::sort(objects.begin(), objects.end(),
                [](HailoDetectionPtr a, HailoDetectionPtr b)
                { return a->get_confidence() > b->get_confidence(); });

        for (uint index = 0; index < objects.size(); index++)
        {
            for (uint jindex = index + 1; jindex < objects.size(); jindex++)
            {
                if (objects[index]->get_class_id() == objects[jindex]->get_class_id())
                {
                    // For each detection, calculate the IOU against each following detection.
                    float iou = iou_calc(objects[index]->get_bbox(), objects[jindex]->get_bbox());
                    // If the IOU is above threshold, then we have two similar detections,
                    // and want to delete the one.
                    if (iou >= iou_thr)
                    {
                        // The detections are arranged in highest score order,
                        // so we want to erase the latter detection.
                        hailo_roi->remove_object(objects[jindex]);
                        objects.erase(objects.begin() + jindex);
                        jindex--; // Step back jindex since we just erased the current detection.
                    }
                }
            }
        }
    }

    void loop() override
    {
        init();

        while (!m_end_of_stream)
        {
            // the first queue (first to subscribe) is the one that is condisidered the "main stream"
            BufferPtr main_buffer = m_queues[0]->pop();
            if (main_buffer == nullptr && m_end_of_stream)
            {
                break;
            }

            // Check if the main buffer has cropping metadata
            int num_subframes = 0; // If no cropping meta provided, then no subframes are requested
            std::vector<MetadataPtr> metadata = main_buffer->get_metadata_of_type(MetadataType::EXPECTED_CROPS);
            if (metadata.size() > 0)
            {
                CroppingMetadataPtr cropping_metadata = std::dynamic_pointer_cast<CroppingMetadata>(metadata[0]);
                num_subframes = cropping_metadata->get_num_crops();
                main_buffer->remove_metadata(cropping_metadata);
            }
            // If no subframes are requested, send the main buffer as is
            if (num_subframes == 0)
            {
                main_buffer->add_time_stamp(m_stage_name);
                set_duration(main_buffer);
                send_to_subscribers(main_buffer);
                continue;
            }

            //Check how many sub frames are available, wait if blocking is enabled
            std::vector<BufferPtr> subframes;
            if (m_blocking)
            {
                for (int i = 0; i < num_subframes; i++)
                {
                    subframes.push_back(m_queues[1]->pop());
                    if (subframes[i] == nullptr && m_end_of_stream)
                    {
                        deinit();
                        return;
                    }
                }
            }
            else 
            {
                // leaky case
                if (m_queues[1]->size() >= num_subframes)
                {
                    for (int i = 0; i < num_subframes; i++)
                    {
                        subframes.push_back(m_queues[1]->pop());
                        if (subframes[i] == nullptr && m_end_of_stream)
                        {
                            deinit();
                            return;
                        }
                    }
                } else {
                    // if not blocking and not enough subframes, then continue
                    main_buffer->add_time_stamp(m_stage_name);
                    set_duration(main_buffer);
                    send_to_subscribers(main_buffer);
                    continue;
                }
            }

            if (m_print_fps && !m_first_fps_measured)
            {
                m_last_time = std::chrono::steady_clock::now();
                m_first_fps_measured = true;
            }

            // copy metadata from subframes to main frame
            //for (auto BufferPtr subframe : subframes)
            for (BufferPtr subframe : subframes)
            {
                if (m_multi_scale)
                {
                    remove_exceeded_bboxes(subframe->get_roi(), m_border_threshold);
                }
                // Flatten subframe roi detections to main_buffer roi's scales.
                // Passing HAILO_DETECTION as a filter type here request to flatten only HailoDetection objects.
                // This passes ownership of the rois to the main buffer
                flatten_hailo_roi(subframe->get_roi(), main_buffer->get_roi(), HAILO_DETECTION);
            }

            if (m_multi_scale)
            {
                // Perform NMS on the main frame's detections after aggragation is done
                nms(main_buffer->get_roi(), m_iou_threshold);
            }

            main_buffer->add_time_stamp(m_stage_name);
            // pass the main_buffer to the subscribers
            set_duration(main_buffer);
            send_to_subscribers(main_buffer);

            if (m_print_fps)
            {
                m_counter++;
                print_fps();
            }
        }
    
        deinit();
    }

};