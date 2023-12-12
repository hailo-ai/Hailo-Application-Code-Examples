/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#ifndef _EXAMPLE_YOLOV8_H_
#define _EXAMPLE_YOLOV8_H_

#pragma once
#include "media_library/common/common.h"
#include <hailo/hailort.hpp>
#include <hailo/vdevice.hpp>
#include <hailo/infer_model.hpp>
#include <hailo/vstream.hpp>
#include "media_library/buffer_pool.hpp"
#include <queue>
#include <semaphore>
#include <boost/thread.hpp>
#include <boost/lockfree/queue.hpp>
#include <future>
#include <atomic>

class Yolov8{
    private:
    std::unique_ptr<hailort::VDevice> vdevice;
    const int kQueueCapacity = 20; 
    std::atomic<bool> stopSignal;
    std::future<hailo_status> write_thread;
    std::future<hailo_status> read_thread;
    std::future<hailo_status> pp_thread;
    boost::lockfree::queue<hailo_media_library_buffer*> input_write_queue;
    boost::lockfree::queue<hailo_media_library_buffer*> write_read_queue;
    boost::lockfree::queue<hailo_media_library_buffer*> read_pp_queue;
    boost::lockfree::queue<hailo_media_library_buffer*> pp_output_queue;
    std::vector<std::shared_ptr<FeatureData>> features;
    std::shared_ptr<hailort::ConfiguredNetworkGroup> network_group;
    std::shared_ptr<std::pair<std::vector<hailort::InputVStream>,std::vector<hailort::OutputVStream>>> vstreams;
    int width;
    int height;
    hailo_status write();
    hailo_status post_process(std::vector<std::shared_ptr<FeatureData>> &features);
    hailo_status read(std::vector<std::shared_ptr<FeatureData>> features);
    public:
    Yolov8(std::string hef, int width, int height);
    hailo_status run();
    void add_frame(hailo_media_library_buffer& buffer);
    void stop();
    boost::lockfree::queue<hailo_media_library_buffer*>* get_queue();
};



#endif /* _EXAMPLE_YOLOV8_H_ */  