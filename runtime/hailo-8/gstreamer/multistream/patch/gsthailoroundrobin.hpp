/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
/*
 * GStreamer ROUND_ROBIN element
 *
 *
 * gsthailoroundrobin.hpp: Simple ROUND_ROBIN (N->1) element, waits on all sinks, passes the first one, with all metadata included.
 */

#pragma once

#include <gst/gst.h>
#include <vector>
#include <queue>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <pthread.h>
#include <thread>

G_BEGIN_DECLS

#define GST_TYPE_HAILO_ROUND_ROBIN \
    (gst_hailo_round_robin_get_type())
#define GST_HAILO_ROUND_ROBIN(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_HAILO_ROUND_ROBIN, GstHailoRoundRobin))
#define GST_HAILO_ROUND_ROBIN_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_HAILO_ROUND_ROBIN, GstHailoRoundRobinClass))
#define GST_IS_HAILO_ROUND_ROBIN(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_HAILO_ROUND_ROBIN))
#define GST_IS_HAILO_ROUND_ROBIN_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_HAILO_ROUND_ROBIN))
#define GST_HAILO_ROUND_ROBIN_CAST(obj) ((GstHailoRoundRobin *)(obj))

typedef struct _GstHailoRoundRobin GstHailoRoundRobin;
typedef struct _GstHailoRoundRobinClass GstHailoRoundRobinClass;

typedef enum
{
    GST_HAILO_ROUND_ROBIN_MODE_FUNNEL_MODE = 0,
    GST_HAILO_ROUND_ROBIN_MODE_BLOCKING = 1,
    GST_HAILO_ROUND_ROBIN_MODE_NON_BLOCKING = 2,
} GstHailoRoundRobinMode;

/**
 * GstHailoRoundRobin:
 *
 * Opaque #GstHailoRoundRobin data structure.
 */
struct _GstHailoRoundRobin
{
    GstElement element;
    GstPad *srcpad;
    size_t current_pad_num;
    GstHailoRoundRobinMode mode;
    uint retries_num;
    uint queue_size;
    uint wait_time;
    uint preroll_frames;
    std::vector<std::unique_ptr<std::mutex>> mutexes_blocking;
    std::vector<std::unique_ptr<std::mutex>> mutexes_non_blocking;
    std::unique_ptr<std::shared_mutex> counter_mutex;
    int buffer_counter;
    std::vector<std::unique_ptr<std::condition_variable>> condition_vars_blocking;
    std::vector<std::unique_ptr<std::condition_variable>> condition_vars_non_blocking;
    std::vector<std::unique_ptr<std::queue<GstBuffer *>>> pad_queues;
    std::thread *thread;
    gboolean stop_thread;
    std::unique_ptr<std::shared_mutex> current_pad_mutex;
};

struct _GstHailoRoundRobinClass
{
    GstElementClass parent_class;
};

G_GNUC_INTERNAL GType gst_hailo_round_robin_get_type(void);

G_END_DECLS
