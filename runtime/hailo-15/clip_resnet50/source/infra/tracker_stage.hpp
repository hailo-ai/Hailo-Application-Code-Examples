#pragma once
#include "stage.hpp"
#include "buffer.hpp"
#include "queue.hpp"
#include "hailo_tracker.hpp"


class TrackerStage : public ConnectedStage
{
private:
    std::string m_tracker_name = "hailo_tracker";
    HailoTrackerParams m_tracker_params;
    int m_class_id;
public:
    TrackerStage(std::string name, size_t queue_size=5, bool leaky=false, int classification_id=-1, bool print_fps=false) : 
        ConnectedStage(name, queue_size, leaky, print_fps), m_class_id(classification_id){}

    AppStatus init() override
    {
        m_tracker_params.kalman_distance = DEFAULT_KALMAN_DISTANCE;
        m_tracker_params.iou_threshold = DEFAULT_IOU_THRESHOLD;
        m_tracker_params.init_iou_threshold = DEFAULT_INIT_IOU_THRESHOLD;
        m_tracker_params.keep_tracked_frames = DEFAULT_KEEP_FRAMES;
        m_tracker_params.keep_new_frames = DEFAULT_KEEP_FRAMES;
        m_tracker_params.keep_lost_frames = DEFAULT_KEEP_FRAMES;
        m_tracker_params.keep_past_metadata = DEFAULT_KEEP_PAST_METADATA;
        m_tracker_params.std_weight_position = DEFAULT_STD_WEIGHT_POSITION;
        m_tracker_params.std_weight_position_box = DEFAULT_STD_WEIGHT_POSITION_BOX;
        m_tracker_params.std_weight_velocity = DEFAULT_STD_WEIGHT_VELOCITY;
        m_tracker_params.std_weight_velocity_box = DEFAULT_STD_WEIGHT_VELOCITY_BOX;
        m_tracker_params.debug = DEFAULT_DEBUG;
        m_tracker_params.hailo_objects_blacklist = DEFAULT_HAILO_OBJECTS_BLACKLIST;
        return AppStatus::SUCCESS;
    }

    AppStatus deinit() override
    {
        return AppStatus::SUCCESS;
    }

    AppStatus process(BufferPtr data)
    {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        HailoROIPtr hailo_roi = data->get_roi();

        std::vector<HailoDetectionPtr> detections;
        for (auto obj : hailo_roi->get_objects_typed(HAILO_DETECTION))
        {
            HailoDetectionPtr detection = std::dynamic_pointer_cast<HailoDetection>(obj);
            if ((m_class_id == -1) || (detection->get_class_id() == m_class_id))
            {
                detections.push_back(detection);
                hailo_roi->remove_object(detection);
            }
        }

        // Swap the detections in the roi with just the online tracked detections
        std::vector<HailoDetectionPtr> online_detection_ptrs = HailoTracker::GetInstance().update(m_tracker_name, detections);

        hailo_common::add_detection_pointers(hailo_roi, online_detection_ptrs);
        
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if (m_print_fps)
        {
            std::cout << "Tracker time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]" << std::endl;
        }
        data->add_time_stamp(m_stage_name);
        set_duration(data);
        send_to_subscribers(data);

        return AppStatus::SUCCESS;
    }
};