/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
/*
 * GStreamer RoundRobin element
 *
 * gsthailoroundrobin.cpp: Simple Input Round Robin funnel (N->1) element, waits on all sinks, passes the first one, with all metadata included.
 */

#include "gsthailoroundrobin.hpp"
#include "gst_hailo_meta.hpp"
#include "gst_hailo_stream_meta.hpp"

GST_DEBUG_CATEGORY_STATIC(gst_hailo_round_robin_debug);
#define GST_CAT_DEFAULT gst_hailo_round_robin_debug

GType gst_hailo_round_robin_pad_get_type(void);
#define GST_TYPE_HAILO_ROUND_ROBIN_PAD \
    (gst_hailo_round_robin_pad_get_type())
#define GST_HAILO_ROUND_ROBIN_PAD(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_HAILO_ROUND_ROBIN_PAD, GstHailoRoundRobinPad))
#define GST_HAILO_ROUND_ROBIN_PAD_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_HAILO_ROUND_ROBIN_PAD, GstHailoRoundRobinPadClass))
#define GST_IS_ROUND_ROBIN_PAD(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_HAILO_ROUND_ROBIN_PAD))
#define GST_IS_ROUND_ROBIN_PAD_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_HAILO_ROUND_ROBIN_PAD))
#define GST_HAILO_ROUND_ROBIN_PAD_CAST(obj) \
    ((GstHailoRoundRobinPad *)(obj))

typedef struct _GstHailoRoundRobinPad GstHailoRoundRobinPad;
typedef struct _GstHailoRoundRobinPadClass GstHailoRoundRobinPadClass;

#define GST_TYPE_HAILOROUNDROBIN_MODE (gst_hailoroundrobin_mode_get_type())
static GType
gst_hailoroundrobin_mode_get_type(void)
{
    static GType hailoroundrobin_mode_type = 0;
    static const GEnumValue hailoroundrobin_modes[] = {
        {GST_HAILO_ROUND_ROBIN_MODE_FUNNEL_MODE, "Funnel Mode (push every buffer when it is ready)", "funnel-mode"},
        {GST_HAILO_ROUND_ROBIN_MODE_BLOCKING, "Blocking Mode (push every buffer when it is its pad's turn, and if the buffer is not ready, block until ready)", "blocking-mode"},
        {GST_HAILO_ROUND_ROBIN_MODE_NON_BLOCKING, "Non Blocking Mode (push every buffer when it is its pad's turn, and if the buffer is not ready, skip it)", "non-blocking-mode"},
        {0, NULL, NULL},
    };
    if (!hailoroundrobin_mode_type)
    {
        hailoroundrobin_mode_type =
            g_enum_register_static("GstHailoRoundRobinMode", hailoroundrobin_modes);
    }
    return hailoroundrobin_mode_type;
}

struct _GstHailoRoundRobinPad
{
    GstPad parent;
    gboolean got_eos;
};

struct _GstHailoRoundRobinPadClass
{
    GstPadClass parent;
};

static size_t get_pad_num(GstPad *pad)
{
    gchar *name = gst_pad_get_name(pad);
    std::string pad_name_str(name);
    std::string pad_num_str = pad_name_str.substr(pad_name_str.find_first_of("0123456789"));
    size_t pad_num = stoi(pad_num_str);
    g_free(name);
    return pad_num;
}

G_DEFINE_TYPE(GstHailoRoundRobinPad, gst_hailo_round_robin_pad, GST_TYPE_PAD);

#define DEFAULT_FORWARD_STICKY_EVENTS TRUE

enum
{
    PROP_0,
    PROP_MODE,
    PROP_RETRIES_NUM,
    PROP_QUEUE_SIZE,
    PROP_WAIT_TIME,
    PROP_PREROLL_FRAMES,
};

static void
gst_hailo_round_robin_pad_class_init(GstHailoRoundRobinPadClass *klass)
{
}

static void
gst_hailo_round_robin_pad_init(GstHailoRoundRobinPad *pad)
{
    pad->got_eos = FALSE;
}

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE("sink_%u",
                                                                    GST_PAD_SINK,
                                                                    GST_PAD_REQUEST,
                                                                    GST_STATIC_CAPS_ANY);

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE("src",
                                                                   GST_PAD_SRC,
                                                                   GST_PAD_ALWAYS,
                                                                   GST_STATIC_CAPS_ANY);

#define _do_init \
    GST_DEBUG_CATEGORY_INIT(gst_hailo_round_robin_debug, "hailo_round_robin", 0, "hailo_round_robin element");
#define gst_hailo_round_robin_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE(GstHailoRoundRobin, gst_hailo_round_robin, GST_TYPE_ELEMENT, _do_init);

static GstStateChangeReturn gst_hailo_round_robin_change_state(GstElement *element,
                                                               GstStateChange transition);

static GstFlowReturn gst_hailo_round_robin_sink_chain_preroll(GstPad *pad,
                                                              GstObject *parent,
                                                              GstBuffer *buf);
static GstFlowReturn gst_hailo_round_robin_sink_chain_funnel_mode(GstPad *pad,
                                                                  GstObject *parent,
                                                                  GstBuffer *buf);
static GstFlowReturn gst_hailo_round_robin_sink_chain_blocking_mode(GstPad *pad,
                                                                    GstObject *parent,
                                                                    GstBuffer *buf);
static GstFlowReturn gst_hailo_round_robin_sink_chain_non_blocking_mode(GstPad *pad,
                                                                        GstObject *parent,
                                                                        GstBuffer *buf);

static gboolean gst_hailo_round_robin_sink_event(GstPad *pad,
                                                 GstObject *parent,
                                                 GstEvent *event);

static GstPad *gst_hailo_round_robin_request_new_pad(GstElement *element,
                                                     GstPadTemplate *templ,
                                                     const gchar *name,
                                                     const GstCaps *caps);

static void gst_hailo_round_robin_release_pad(GstElement *element, GstPad *pad);
static void gst_hailo_round_robin_dispose(GObject *object);

static void
gst_hailo_round_robin_set_property(GObject *object, guint prop_id,
                                   const GValue *value, GParamSpec *pspec)
{
    switch (prop_id)
    {
    case PROP_MODE:
    {
        GST_HAILO_ROUND_ROBIN(object)->mode = (GstHailoRoundRobinMode)g_value_get_enum(value);
        break;
    }
    case PROP_RETRIES_NUM:
    {
        GST_HAILO_ROUND_ROBIN(object)->retries_num = g_value_get_uint(value);
        break;
    }
    case PROP_QUEUE_SIZE:
    {
        GST_HAILO_ROUND_ROBIN(object)->queue_size = g_value_get_uint(value);
        break;
    }
    case PROP_WAIT_TIME:
    {
        GST_HAILO_ROUND_ROBIN(object)->wait_time = g_value_get_uint(value);
        break;
    }
    case PROP_PREROLL_FRAMES:
    {
        GST_HAILO_ROUND_ROBIN(object)->preroll_frames = g_value_get_uint(value);
        break;
    }
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void
gst_hailo_round_robin_get_property(GObject *object, guint prop_id, GValue *value,
                                   GParamSpec *pspec)
{
    switch (prop_id)
    {
    case PROP_MODE:
        g_value_set_enum(value, GST_HAILO_ROUND_ROBIN(object)->mode);
        break;
    case PROP_RETRIES_NUM:
        g_value_set_uint(value, GST_HAILO_ROUND_ROBIN(object)->retries_num);
        break;
    case PROP_QUEUE_SIZE:
        g_value_set_uint(value, GST_HAILO_ROUND_ROBIN(object)->queue_size);
        break;
    case PROP_WAIT_TIME:
        g_value_set_uint(value, GST_HAILO_ROUND_ROBIN(object)->wait_time);
        break;
    case PROP_PREROLL_FRAMES:
        g_value_set_uint(value, GST_HAILO_ROUND_ROBIN(object)->preroll_frames);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

size_t get_current_pad_num(GstHailoRoundRobin *hailo_round_robin)
{
    std::shared_lock lock(*(hailo_round_robin->current_pad_mutex.get())); // reader lock
    return hailo_round_robin->current_pad_num;
}

void set_current_pad_num(GstHailoRoundRobin *hailo_round_robin, size_t num)
{
    std::unique_lock lock(*(hailo_round_robin->current_pad_mutex.get())); // writer lock
    hailo_round_robin->current_pad_num = num;
}

int get_buffer_counter_value(GstHailoRoundRobin *hailo_round_robin)
{
    std::shared_lock lock(*(hailo_round_robin->counter_mutex.get())); // reader lock
    return hailo_round_robin->buffer_counter;
}

void set_buffer_counter_value(GstHailoRoundRobin *hailo_round_robin, size_t val)
{
    std::unique_lock lock(*(hailo_round_robin->counter_mutex.get())); // writer lock
    hailo_round_robin->buffer_counter = val;
}

void increment_buffer_counter_value(GstHailoRoundRobin *hailo_round_robin)
{
    std::unique_lock lock(*(hailo_round_robin->counter_mutex.get())); // writer lock
    if (hailo_round_robin->buffer_counter != -1)
        hailo_round_robin->buffer_counter++;
}

static gboolean
forward_events(GstPad *pad, GstEvent **event, gpointer user_data)
{
    gboolean res = TRUE;
    // This function pushes forward all events that are not EOS.
    GstPad *srcpad = GST_PAD_CAST(user_data);

    if (GST_EVENT_TYPE(*event) != GST_EVENT_EOS)
    {
        res = gst_pad_push_event(srcpad, gst_event_ref(*event));
    }
    return res;
}

void schedule(GstHailoRoundRobin *hailo_round_robin)
{
    GstFlowReturn res;
    while (!hailo_round_robin->stop_thread)
    {
        // iterate the pad queues and push the first buffer in each queueq
        for (uint i = 0; i < hailo_round_robin->pad_queues.size(); i++)
        {
            if (hailo_round_robin->pad_queues[i] != NULL)
            {
                for (uint j = 0; j <= hailo_round_robin->retries_num; j++)
                {
                    if (hailo_round_robin->pad_queues[i]->empty())
                    {
                        if (j < hailo_round_robin->retries_num) // don't sleep on the last iteration
                        {
                            sleep((float)hailo_round_robin->wait_time / 1000.0);
                        }
                    }
                    else
                    {
                        set_current_pad_num(hailo_round_robin, i);
                        GstBuffer *buf = hailo_round_robin->pad_queues[i]->front();
                        hailo_round_robin->pad_queues[i]->pop();
                        if (hailo_round_robin->condition_vars_non_blocking[i] == NULL)
                        {
                            continue;
                        }
                        hailo_round_robin->condition_vars_non_blocking[i]->notify_one();

                        GstPad *pad = gst_element_get_static_pad(GST_ELEMENT_CAST(hailo_round_robin), ("sink_" + std::to_string(i)).c_str());

                        if (pad == NULL) // not found, will try different method - maybe the pad names are hailoroundrobinpad0, hailoroundrobinpad1, etc.
                        {
                            pad = gst_element_get_static_pad(GST_ELEMENT_CAST(hailo_round_robin), ("hailoroundrobinpad" + std::to_string(i)).c_str());
                            if (pad == NULL)
                            {
                                GST_ERROR_OBJECT(hailo_round_robin, "Failed to get pad %d", i);
                                continue;
                            }
                        }
                        gchar *pad_name = gst_pad_get_name(pad);
                        gchar *stream_id = gst_pad_get_stream_id(pad);

                        // Add stream meta to the buffer including the pad name and stream id.
                        gst_buffer_add_hailo_stream_meta(buf, pad_name, stream_id);

                        // Forward sticky events.
                        gst_pad_sticky_events_foreach(pad, forward_events, hailo_round_robin->srcpad);

                        // Push out_buffer forward.
                        res = gst_pad_push(hailo_round_robin->srcpad, buf);
                        if (res != GST_FLOW_OK)
                        {
                            GST_ERROR_OBJECT(hailo_round_robin, "Failed to push buffer to srcpad");
                        }
                        gst_object_unref(pad);
                        g_free(pad_name);
                        g_free(stream_id);
                        break; // make sure it only pushes 1 buffer per pad
                    }
                }
            }
        }
    }
}

static void
gst_hailo_round_robin_class_init(GstHailoRoundRobinClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass *gstelement_class = GST_ELEMENT_CLASS(klass);

    gobject_class->set_property = gst_hailo_round_robin_set_property;
    gobject_class->get_property = gst_hailo_round_robin_get_property;
    gobject_class->dispose = GST_DEBUG_FUNCPTR(gst_hailo_round_robin_dispose);

    gst_element_class_set_static_metadata(gstelement_class,
                                          "Input Round Robin element", "Generic", "multiple input into one output in roundrobin order",
                                          "hailo.ai <contact@hailo.ai>");

    gst_element_class_add_static_pad_template(gstelement_class, &sink_template);
    gst_element_class_add_static_pad_template(gstelement_class, &src_template);

    gstelement_class->request_new_pad =
        GST_DEBUG_FUNCPTR(gst_hailo_round_robin_request_new_pad);
    gstelement_class->release_pad = GST_DEBUG_FUNCPTR(gst_hailo_round_robin_release_pad);
    gstelement_class->change_state = GST_DEBUG_FUNCPTR(gst_hailo_round_robin_change_state);

    // install new property mode
    g_object_class_install_property(gobject_class,
                                    PROP_MODE,
                                    g_param_spec_enum("mode",
                                                      "mode",
                                                      "Select the mode of the element (0 - funnel mode (push every buffer when it is ready), 1 - blocking mode (push every buffer when it is its pad's turn, and if the buffer is not ready, block until ready), 2 - non blocking mode(push every buffer when it is its pad's turn, and if the buffer is not ready, skip it))",
                                                      GST_TYPE_HAILOROUNDROBIN_MODE,
                                                      (gint)GST_HAILO_ROUND_ROBIN_MODE_BLOCKING,
                                                      (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class,
                                    PROP_RETRIES_NUM,
                                    g_param_spec_uint("retries-num",
                                                      "Retries num",
                                                      "Number of retries to get a buffer from a pad queue (only relevant when using non-blocking mode)",
                                                      1,
                                                      20,
                                                      1,
                                                      (GParamFlags)(GST_PARAM_CONTROLLABLE | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class,
                                    PROP_QUEUE_SIZE,
                                    g_param_spec_uint("queue-size",
                                                      "Queue size",
                                                      "Size of the queue for each pad (only relevant when using non-blocking mode)",
                                                      1,
                                                      10,
                                                      3,
                                                      (GParamFlags)(GST_PARAM_CONTROLLABLE | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class,
                                    PROP_WAIT_TIME,
                                    g_param_spec_uint("wait-time",
                                                      "Wait Time",
                                                      "Time in ms to wait between tries to get a buffer from a pad queue (only relevant when using non-blocking mode)",
                                                      1,
                                                      500,
                                                      5,
                                                      (GParamFlags)(GST_PARAM_CONTROLLABLE | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class,
                                    PROP_PREROLL_FRAMES,
                                    g_param_spec_uint("preroll-frames",
                                                      "Preroll frames",
                                                      "Number of frames to operate in blocking mode before moving to non-blocking-mode (only relevant when using non-blocking mode)",
                                                      1,
                                                      30,
                                                      3,
                                                      (GParamFlags)(GST_PARAM_CONTROLLABLE | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
}

static void
gst_hailo_round_robin_init(GstHailoRoundRobin *hailo_round_robin)
{
    hailo_round_robin->current_pad_num = 0;
    hailo_round_robin->mutexes_blocking.clear();
    hailo_round_robin->mutexes_non_blocking.clear();
    hailo_round_robin->pad_queues.clear();
    hailo_round_robin->condition_vars_blocking.clear();
    hailo_round_robin->condition_vars_non_blocking.clear();
    hailo_round_robin->buffer_counter = 0;
    hailo_round_robin->retries_num = 1;
    hailo_round_robin->queue_size = 3;
    hailo_round_robin->wait_time = 0;
    hailo_round_robin->preroll_frames = 3;
    hailo_round_robin->stop_thread = false;
    hailo_round_robin->srcpad = gst_pad_new_from_static_template(&src_template, "src");
    hailo_round_robin->mode = GST_HAILO_ROUND_ROBIN_MODE_BLOCKING;
    hailo_round_robin->current_pad_mutex = std::make_unique<std::shared_mutex>();
    hailo_round_robin->counter_mutex = std::make_unique<std::shared_mutex>();
    gst_pad_use_fixed_caps(hailo_round_robin->srcpad);
    gst_element_add_pad(GST_ELEMENT(hailo_round_robin), hailo_round_robin->srcpad);
}

static void
gst_hailo_round_robin_dispose(GObject *object)
{
    GstHailoRoundRobin *hailo_round_robin = GST_HAILO_ROUND_ROBIN_CAST(object);
    hailo_round_robin->srcpad = NULL;
    hailo_round_robin->current_pad_num = 0;
    hailo_round_robin->pad_queues.clear();
    hailo_round_robin->buffer_counter = 0;
    hailo_round_robin->mutexes_blocking.clear();
    hailo_round_robin->mutexes_non_blocking.clear();
    hailo_round_robin->condition_vars_blocking.clear();
    hailo_round_robin->condition_vars_non_blocking.clear();
    G_OBJECT_CLASS(parent_class)->dispose(object);
}

void set_chain_to_all_pads(GstHailoRoundRobin *hailo_round_robin, GstPadChainFunction chain_function)
{
    GstElement *element = GST_ELEMENT_CAST(hailo_round_robin);
    GList *item;
    for (item = element->sinkpads; item != NULL; item = g_list_next(item))
    {
        GstPad *sinkpad = GST_PAD_CAST(item->data);
        gst_pad_set_chain_function(sinkpad, GST_DEBUG_FUNCPTR(chain_function));
    }
}

static GstPad *
gst_hailo_round_robin_request_new_pad(GstElement *element, GstPadTemplate *templ,
                                      const gchar *name, const GstCaps *caps)
{
    GstPad *sinkpad;
    GstHailoRoundRobin *hailo_round_robin = GST_HAILO_ROUND_ROBIN_CAST(element);
    GST_DEBUG_OBJECT(element, "requesting pad");

    sinkpad = GST_PAD_CAST(g_object_new(GST_TYPE_HAILO_ROUND_ROBIN_PAD,
                                        "name", name, "direction", templ->direction, "template", templ,
                                        NULL));

    // Set sink_chain and sink_event funtions for the new pad
    switch (hailo_round_robin->mode)
    {
    case GST_HAILO_ROUND_ROBIN_MODE_FUNNEL_MODE:
    {
        gst_pad_set_chain_function(sinkpad, GST_DEBUG_FUNCPTR(gst_hailo_round_robin_sink_chain_funnel_mode));
        break;
    }
    case GST_HAILO_ROUND_ROBIN_MODE_BLOCKING:
    {
        gst_pad_set_chain_function(sinkpad, GST_DEBUG_FUNCPTR(gst_hailo_round_robin_sink_chain_blocking_mode));
        break;
    }
    case GST_HAILO_ROUND_ROBIN_MODE_NON_BLOCKING:
    {
        gst_pad_set_chain_function(sinkpad, GST_DEBUG_FUNCPTR(gst_hailo_round_robin_sink_chain_preroll)); // this will be changed to non-blocking mode after specific number of buffers
        break;
    }
    }

    gst_pad_set_event_function(sinkpad,
                               GST_DEBUG_FUNCPTR(gst_hailo_round_robin_sink_event));

    GST_OBJECT_FLAG_SET(sinkpad, GST_PAD_FLAG_PROXY_CAPS);
    GST_OBJECT_FLAG_SET(sinkpad, GST_PAD_FLAG_PROXY_ALLOCATION);

    hailo_round_robin->mutexes_blocking.emplace_back(std::make_unique<std::mutex>());
    hailo_round_robin->mutexes_non_blocking.emplace_back(std::make_unique<std::mutex>());
    hailo_round_robin->pad_queues.emplace_back(std::make_unique<std::queue<GstBuffer *>>()); // create a new queue for the new pad
    hailo_round_robin->condition_vars_blocking.emplace_back(std::make_unique<std::condition_variable>());
    hailo_round_robin->condition_vars_non_blocking.emplace_back(std::make_unique<std::condition_variable>());

    gst_pad_set_active(sinkpad, TRUE);

    gst_element_add_pad(element, sinkpad);

    GST_DEBUG_OBJECT(element, "requested pad %s:%s",
                     GST_DEBUG_PAD_NAME(sinkpad));

    return sinkpad;
}

static gboolean
gst_hailo_round_robin_all_sinkpads_eos_unlocked(GstHailoRoundRobin *hailo_round_robin)
{
    GstElement *element = GST_ELEMENT_CAST(hailo_round_robin);
    GList *item;
    gboolean all_eos = FALSE;

    // When there are no sinkpads we return FALSE because there is no need to send EOS down the pipeline, pipeline is still active.
    if (element->numsinkpads == 0)
    {
        goto done;
    }
    // Iterate over all the sinkpads of the element, and check if they are in eos state
    for (item = element->sinkpads; item != NULL; item = g_list_next(item))
    {
        GstHailoRoundRobinPad *sinkpad = GST_HAILO_ROUND_ROBIN_PAD_CAST(item->data);

        if (!sinkpad->got_eos)
        {
            return FALSE;
        }
    }

    all_eos = TRUE;

done:
    return all_eos;
}

static void
gst_hailo_round_robin_release_pad(GstElement *element, GstPad *pad)
{
    GstHailoRoundRobin *hailo_round_robin = GST_HAILO_ROUND_ROBIN_CAST(element);
    GST_DEBUG_OBJECT(hailo_round_robin, "releasing pad %s:%s", GST_DEBUG_PAD_NAME(pad));
    gst_pad_set_active(pad, FALSE);

    if (hailo_round_robin->condition_vars_blocking[get_pad_num(pad)] != NULL)
        hailo_round_robin->condition_vars_blocking[get_pad_num(pad)]->notify_all();

    if (hailo_round_robin->condition_vars_non_blocking[get_pad_num(pad)] != NULL)
        hailo_round_robin->condition_vars_non_blocking[get_pad_num(pad)]->notify_all();
    gst_element_remove_pad(GST_ELEMENT_CAST(hailo_round_robin), pad);
}

static GstFlowReturn
gst_hailo_round_robin_sink_chain_preroll(GstPad *pad, GstObject *parent, GstBuffer *buf)
{
    GstFlowReturn ret = GST_FLOW_ERROR;
    GstHailoRoundRobin *hailo_round_robin = GST_HAILO_ROUND_ROBIN_CAST(parent);

    size_t pad_num = get_pad_num(pad);
    if (hailo_round_robin->current_pad_num != pad_num)
    {
        // Wait for the turn of this pad.
        if (hailo_round_robin->condition_vars_blocking[pad_num] != NULL)
        {
            std::unique_lock lock(*hailo_round_robin->mutexes_blocking[pad_num].get());
            hailo_round_robin->condition_vars_blocking[pad_num]->wait(lock);
        }
        else
        {
            gst_buffer_unref(buf);
            return ret;
        }

        // If condition variable got notified but it is not this pad's turn, raise an error.
        if ((hailo_round_robin->current_pad_num != pad_num) && (get_buffer_counter_value(hailo_round_robin) != -1))
        {
            GST_ERROR_OBJECT(hailo_round_robin,
                             "Tried to send buf on pad %zu while current pad should be %zu, dropping buffer!",
                             pad_num, hailo_round_robin->current_pad_num);
            gst_buffer_unref(buf);
            return ret;
        }
    }

    buf = gst_buffer_make_writable(buf);

    gchar *pad_name = gst_pad_get_name(pad);
    gchar *stream_id = gst_pad_get_stream_id(pad);

    // Add stream meta to the buffer including the pad name and stream id.
    gst_buffer_add_hailo_stream_meta(buf, pad_name, stream_id);

    // Forward sticky events.
    gst_pad_sticky_events_foreach(pad, forward_events, hailo_round_robin->srcpad);

    // Push out_buffer forward.
    ret = gst_pad_push(hailo_round_robin->srcpad, buf);

    increment_buffer_counter_value(hailo_round_robin); // increment only if not equal to -1

    if (get_buffer_counter_value(hailo_round_robin) == hailo_round_robin->mutexes_blocking.size() * hailo_round_robin->preroll_frames) // 10 buffers per pad
    {
        set_buffer_counter_value(hailo_round_robin, -1); // don't use it anymore
        std::thread schedule_thread(schedule, hailo_round_robin);
        hailo_round_robin->thread = std::move(&schedule_thread);
        hailo_round_robin->thread->detach();
        set_chain_to_all_pads(hailo_round_robin, gst_hailo_round_robin_sink_chain_non_blocking_mode);
        hailo_round_robin->mode = GST_HAILO_ROUND_ROBIN_MODE_NON_BLOCKING;
        hailo_round_robin->current_pad_num = 0;

        // notify all pads that are waiting in the blocking mode
        for (size_t i = 0; i < hailo_round_robin->mutexes_blocking.size(); i++)
        {
            if (hailo_round_robin->condition_vars_blocking[i] != NULL)
                hailo_round_robin->condition_vars_blocking[i]->notify_all();
        }
    }

    g_free(pad_name);
    g_free(stream_id);

    if (get_buffer_counter_value(hailo_round_robin) != -1)
    {
        hailo_round_robin->current_pad_num++;
        if (hailo_round_robin->current_pad_num == hailo_round_robin->mutexes_blocking.size())
            hailo_round_robin->current_pad_num = 0;

        while (hailo_round_robin->condition_vars_blocking[hailo_round_robin->current_pad_num] == NULL)
        {
            hailo_round_robin->current_pad_num++;
            if (hailo_round_robin->current_pad_num == hailo_round_robin->mutexes_blocking.size())
                hailo_round_robin->current_pad_num = 0;
        }
        hailo_round_robin->condition_vars_blocking[hailo_round_robin->current_pad_num]->notify_one();
    }
    return ret;
}

static GstFlowReturn
gst_hailo_round_robin_sink_chain_blocking_mode(GstPad *pad, GstObject *parent, GstBuffer *buf)
{
    GstFlowReturn ret = GST_FLOW_ERROR;
    GstHailoRoundRobin *hailo_round_robin = GST_HAILO_ROUND_ROBIN_CAST(parent);

    size_t pad_num = get_pad_num(pad);
    if (hailo_round_robin->current_pad_num != pad_num)
    {
        // Wait for the turn of this pad.
        if (hailo_round_robin->condition_vars_blocking[pad_num] != NULL)
        {
            std::unique_lock lock(*hailo_round_robin->mutexes_blocking[pad_num].get());
            hailo_round_robin->condition_vars_blocking[pad_num]->wait(lock);
        }
        else
        {
            gst_buffer_unref(buf);
            return ret;
        }

        // If condition variable got notified but it is not this pad's turn, raise an error.
        if (hailo_round_robin->current_pad_num != pad_num)
        {
            GST_ERROR_OBJECT(hailo_round_robin,
                             "Tried to send buf on pad %zu while current pad should be %zu, dropping buffer!",
                             pad_num, hailo_round_robin->current_pad_num);
            gst_buffer_unref(buf);
            return ret;
        }
    }

    buf = gst_buffer_make_writable(buf);

    gchar *pad_name = gst_pad_get_name(pad);
    gchar *stream_id = gst_pad_get_stream_id(pad);

    // Add stream meta to the buffer including the pad name and stream id.
    gst_buffer_add_hailo_stream_meta(buf, pad_name, stream_id);

    // Forward sticky events.
    gst_pad_sticky_events_foreach(pad, forward_events, hailo_round_robin->srcpad);

    // Push out_buffer forward.
    ret = gst_pad_push(hailo_round_robin->srcpad, buf);

    g_free(pad_name);
    g_free(stream_id);

    hailo_round_robin->current_pad_num++;
    if (hailo_round_robin->current_pad_num == hailo_round_robin->mutexes_blocking.size())
        hailo_round_robin->current_pad_num = 0;

    while (hailo_round_robin->condition_vars_blocking[hailo_round_robin->current_pad_num] == NULL)
    {
        hailo_round_robin->current_pad_num++;
        if (hailo_round_robin->current_pad_num == hailo_round_robin->mutexes_blocking.size())
            hailo_round_robin->current_pad_num = 0;
    }
    hailo_round_robin->condition_vars_blocking[hailo_round_robin->current_pad_num]->notify_one();
    return ret;
}

static GstFlowReturn
gst_hailo_round_robin_sink_chain_funnel_mode(GstPad *pad, GstObject *parent, GstBuffer *buf)
{
    GstFlowReturn ret = GST_FLOW_ERROR;
    GstHailoRoundRobin *hailo_round_robin = GST_HAILO_ROUND_ROBIN_CAST(parent);
    buf = gst_buffer_make_writable(buf);
    gchar *pad_name = gst_pad_get_name(pad);
    gchar *stream_id = gst_pad_get_stream_id(pad);

    // Add stream meta to the buffer including the pad name and stream id.
    gst_buffer_add_hailo_stream_meta(buf, pad_name, stream_id);

    // Forward sticky events.
    gst_pad_sticky_events_foreach(pad, forward_events, hailo_round_robin->srcpad);

    ret = gst_pad_push(hailo_round_robin->srcpad, buf);
    g_free(pad_name);
    g_free(stream_id);
    return ret;
}

static GstFlowReturn
gst_hailo_round_robin_sink_chain_non_blocking_mode(GstPad *pad, GstObject *parent, GstBuffer *buf)
{
    GstFlowReturn ret = GST_FLOW_ERROR;
    GstHailoRoundRobin *hailo_round_robin = GST_HAILO_ROUND_ROBIN_CAST(parent);
    size_t pad_num = get_pad_num(pad);

    if (hailo_round_robin->condition_vars_non_blocking[pad_num] != NULL)
    {
        std::unique_lock lock(*hailo_round_robin->mutexes_non_blocking[pad_num].get());
        hailo_round_robin->condition_vars_non_blocking[pad_num]->wait(lock, [hailo_round_robin, pad_num]
                                                                      { return hailo_round_robin->pad_queues[pad_num]->size() < hailo_round_robin->queue_size; });

        hailo_round_robin->pad_queues[pad_num]->push(buf);
    }
    else
    {
        gst_buffer_unref(buf);
        return ret;
    }
    ret = GST_FLOW_OK;
    return ret;
}

static gboolean
gst_hailo_round_robin_sink_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
    // Handle sink event comming from the sink pad
    GstHailoRoundRobin *hailo_round_robin = GST_HAILO_ROUND_ROBIN_CAST(parent);
    GstHailoRoundRobinPad *fpad = GST_HAILO_ROUND_ROBIN_PAD_CAST(pad);
    gboolean forward = TRUE;
    gboolean res = TRUE;
    gboolean unlock = FALSE;
    size_t pad_num = get_pad_num(pad);

    GST_DEBUG_OBJECT(pad, "received event %" GST_PTR_FORMAT, event);

    if (GST_EVENT_IS_STICKY(event))
    {
        unlock = TRUE;
        GST_PAD_STREAM_LOCK(hailo_round_robin->srcpad);

        if (GST_EVENT_TYPE(event) == GST_EVENT_EOS)
        {
            GST_OBJECT_LOCK(hailo_round_robin);
            fpad->got_eos = TRUE;
            hailo_round_robin->condition_vars_blocking[pad_num]->notify_all();
            hailo_round_robin->condition_vars_blocking[pad_num] = NULL;
            hailo_round_robin->condition_vars_non_blocking[pad_num]->notify_all();
            hailo_round_robin->condition_vars_non_blocking[pad_num] = NULL;
            forward = gst_hailo_round_robin_all_sinkpads_eos_unlocked(hailo_round_robin);
            GST_OBJECT_UNLOCK(hailo_round_robin);
        }
        else if (pad_num != get_current_pad_num(hailo_round_robin))
        {
            forward = FALSE;
        }
    }
    else if (GST_EVENT_TYPE(event) == GST_EVENT_FLUSH_STOP)
    {
        unlock = TRUE;
        GST_PAD_STREAM_LOCK(hailo_round_robin->srcpad);
        GST_OBJECT_LOCK(hailo_round_robin);
        fpad->got_eos = FALSE;
        GST_OBJECT_UNLOCK(hailo_round_robin);
    }

    if (forward && GST_EVENT_IS_SERIALIZED(event))
    {
        /* If no data is coming and we receive serialized event, need to forward all sticky events.
         * Otherwise downstream has an inconsistent set of sticky events when
         * handling the new event. */
        if (!unlock)
        {
            unlock = TRUE;
            GST_PAD_STREAM_LOCK(hailo_round_robin->srcpad);
        }

        if (pad_num != get_current_pad_num(hailo_round_robin))
        {
            gst_pad_sticky_events_foreach(pad, forward_events, hailo_round_robin->srcpad);
        }
    }

    if (forward)
        res = gst_pad_push_event(hailo_round_robin->srcpad, event);
    else
        gst_event_unref(event);

    if (unlock)
        GST_PAD_STREAM_UNLOCK(hailo_round_robin->srcpad);

    return res;
}

static GstStateChangeReturn
gst_hailo_round_robin_change_state(GstElement *element, GstStateChange transition)
{
    GstStateChangeReturn ret;
    GstHailoRoundRobin *hailo_round_robin = GST_HAILO_ROUND_ROBIN_CAST(element);

    switch (transition)
    {
    case GST_STATE_CHANGE_READY_TO_NULL:
    {
        if (hailo_round_robin->mode != GST_HAILO_ROUND_ROBIN_MODE_FUNNEL_MODE)
        {
            hailo_round_robin->stop_thread = true;
            for (uint i = 0; i < hailo_round_robin->condition_vars_blocking.size(); i++)
            {
                if (hailo_round_robin->condition_vars_blocking[i] != NULL)
                    hailo_round_robin->condition_vars_blocking[i]->notify_all();
            }
            for (uint i = 0; i < hailo_round_robin->condition_vars_non_blocking.size(); i++)
            {
                if (hailo_round_robin->condition_vars_non_blocking[i] != NULL)
                    hailo_round_robin->condition_vars_non_blocking[i]->notify_all();
            }
            break;
        }
    }

    default:
        break;
    }
    ret = GST_ELEMENT_CLASS(parent_class)->change_state(element, transition);

    return ret;
}
