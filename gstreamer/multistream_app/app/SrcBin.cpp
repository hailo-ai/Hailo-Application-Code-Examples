#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <gst/gst.h>
#include <string>

#include "SrcBin.hpp"
#include "hailo_app_useful_funcs.hpp"

GST_DEBUG_CATEGORY(src_bin_debug);
#define GST_CAT_DEFAULT src_bin_debug

guint SrcBin::bin_cnt = 0; // initialize static variable keeping track of number of bins

// Declare a static pad template for the source pad of the bin
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);

// this function is used to seek a video file to the start (used for loopping files)
static gboolean seek_to_start(gpointer data)
{
    GST_DEBUG("Seeking to start...\n");
    GstElement *bin = static_cast<GstElement *>(data);
    // pause the pipeline
    gst_element_set_state(bin, GST_STATE_PAUSED);

    // seek to the start of the file
    gboolean ret = gst_element_seek(bin, 1.0, GST_FORMAT_TIME, GST_SEEK_FLAG_FLUSH, GST_SEEK_TYPE_SET,
                                    0, GST_SEEK_TYPE_NONE, GST_CLOCK_TIME_NONE);
    if (!ret)
    {
        GST_ERROR("Seek ERROR: Failed!\n");
    }
    // resume the pipeline
    gst_element_set_state(bin, GST_STATE_PLAYING);
    return FALSE;
}

// This function is used to restart the RTSP source
// It will move the src_bin to NULL state and then to PLAYING state
static gboolean restart_source(gpointer data)
{
    GST_DEBUG("Restarting source...\n");
    SrcBin *src_bin = static_cast<SrcBin *>(data);
    src_bin->restarting.store(true);
    // pause the pipeline
    gst_element_set_state(src_bin->bin, GST_STATE_NULL);
    // wait for state change to complete
    GstStateChangeReturn ret = gst_element_get_state(src_bin->bin, NULL, NULL, GST_CLOCK_TIME_NONE);
    if (ret == GST_STATE_CHANGE_FAILURE)
    {
        GST_ERROR("ERROR: Failed to set state to NULL\n");
        return true;
    }
    GST_WARNING("Restarting source...\n");
    // resume the pipeline
    gst_element_set_state(src_bin->bin, GST_STATE_PLAYING);
    // wait for state change to complete
    ret = gst_element_get_state(src_bin->bin, NULL, NULL, GST_CLOCK_TIME_NONE);
    // reset watchdog timer
    src_bin->reset_flag.store(true);
    
    if (ret == GST_STATE_CHANGE_FAILURE)
    {
        GST_ERROR("ERROR: Failed to set state to PLAYING\n");
        return true; // keep the timer running
    }
    GST_INFO("Source restarted\n");
    src_bin->restarting.store(false); // clean the restarting flag
    return false; // stop the timer
}

// Pad probe callback to enable looping and masking events
// This callback is connected to the tee sink pad
// It will mask EOS and SEGMENT events and will schedule a seek to the start of the file if EOS is received
// It will also align the buffer PTS to the current base timestamp and will update it on every SEGMENT event
// It is also used to detect if the stream has stalled, a watchdog timer is used to monitor the reset_flag.
static GstPadProbeReturn tee_sink_loop_pad_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer data)
{
    GST_DEBUG("Tee sink pad probe CB\n");
    GstEvent *event = GST_PAD_PROBE_INFO_EVENT(info);
    SrcBin *src_bin = static_cast<SrcBin *>(data);
    // if this is a buffer align the PTS
    if ((info->type & GST_PAD_PROBE_TYPE_BUFFER))
    {
        src_bin->reset_flag.store(true); // We got data, so reset the watchdog
        // Update the buffer PTS
        GST_BUFFER_PTS(GST_BUFFER(info->data)) += src_bin->current_base_timestamp;
        GST_BUFFER_DTS(GST_BUFFER(info->data)) += src_bin->current_base_timestamp;
    }
    // Handling events
    if ((info->type & GST_PAD_PROBE_TYPE_EVENT_BOTH))
    {
        // print event type
        GST_DEBUG("Event type: %s\n", GST_EVENT_TYPE_NAME(event));
        if (src_bin->restarting.load() == true)
        {
            GST_DEBUG("Source is restarting, dropping event\n");
            return GST_PAD_PROBE_DROP;
        }
        // if this is an EOS event
        if (GST_EVENT_TYPE(event) == GST_EVENT_EOS)
        {
            GST_INFO("EOS event received on source %s\n", src_bin->name.c_str());
            // if loop is enabled
            if (src_bin->type == SrcBin::SrcType::URI && src_bin->loop_enable)
            {
                // schedule a seek to the start of the file
                g_timeout_add(1, (GSourceFunc)seek_to_start, src_bin->bin);
            }
            else if (src_bin->type == SrcBin::SrcType::RTSP)
            {
                src_bin->restarting.store(true);
                // schedule a pipeline restart
                g_timeout_add(2000, (GSourceFunc)restart_source, src_bin);
            }
        }
        // if this is a segment event
        if (GST_EVENT_TYPE(event) == GST_EVENT_SEGMENT)
        {
            // get the segment event
            GstSegment *segment;
            gst_event_parse_segment(event, (const GstSegment **)&segment);
            // set the segment base
            segment->base = src_bin->next_base_timestamp;
            src_bin->current_base_timestamp = src_bin->next_base_timestamp;
            src_bin->next_base_timestamp = +segment->stop;
            // print segemnt info
            GST_INFO("Segment event received on source %s\n", src_bin->name.c_str());
            GST_INFO("Segment base: %" GST_TIME_FORMAT "\n", GST_TIME_ARGS(segment->base));
            GST_INFO("Segment start: %" GST_TIME_FORMAT "\n", GST_TIME_ARGS(segment->start));
            GST_INFO("Segment stop: %" GST_TIME_FORMAT "\n", GST_TIME_ARGS(segment->stop));
            GST_INFO("Segment time: %" GST_TIME_FORMAT "\n", GST_TIME_ARGS(segment->time));
            GST_INFO("Segment position: %" GST_TIME_FORMAT "\n", GST_TIME_ARGS(segment->position));
            GST_INFO("Segment duration: %" GST_TIME_FORMAT "\n", GST_TIME_ARGS(segment->duration));
            GST_INFO("Segment format: %s\n", gst_format_get_name(segment->format));
            GST_INFO("Segment flags: %d\n", segment->flags);
            GST_INFO("Segment rate: %f\n", segment->rate);
            GST_INFO("Segment applied_rate: %f\n", segment->applied_rate);
        }
        // mask events so that they are not sent downstream
        switch (GST_EVENT_TYPE(event))
        {
        case GST_EVENT_EOS:
        case GST_EVENT_SEGMENT:
        case GST_EVENT_FLUSH_START:
        case GST_EVENT_FLUSH_STOP:
        case GST_EVENT_QOS:
            return GST_PAD_PROBE_DROP;
        default:
            break;
        }
    }
    return GST_PAD_PROBE_OK;
}

// watchdog thread to monitor if the stream has stalled
// if the stream has stalled, the source will be restarted
// monitoring is done by setting the reset_flag to true on every buffer received
void watchdog_thread(gpointer data, uint32_t timeout = 5)
{
    SrcBin *src_bin = static_cast<SrcBin *>(data);
    GST_WARNING("Watchdog thread started for source %s\n", src_bin->name.c_str());
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::seconds(timeout));
        if (src_bin->restarting.load() == true)
        {
            continue;
        }
        // If the flag hasn't been reset, we assume the stream has stalled
        if (src_bin->reset_flag.exchange(false) == false)
        {
            GST_WARNING("Stream stalled, restarting source\n");
            src_bin->restarting.store(true);
            // schedule a pipeline restart
            g_timeout_add(2000, (GSourceFunc)restart_source, src_bin);
        }
    }
}

// bus sync handler callback
// this callback is used to mask src_bin related bus messages 
static GstBusSyncReply bus_sync_handler(GstBus *bus, GstMessage *message, gpointer data)
{
    GError *error = NULL;
    gchar *debug_info = NULL;
    GST_DEBUG("Message source: %s, type: %s\n", GST_OBJECT_NAME(message->src), GST_MESSAGE_TYPE_NAME(message));
    switch (GST_MESSAGE_TYPE(message))
    {
    case GST_MESSAGE_EOS:
        GST_INFO("EOS event received source: %s\n", GST_OBJECT_NAME(message->src));
        // unref the message
        gst_message_unref(message);
        // not passing the message downstream
        return GST_BUS_DROP;
    case GST_MESSAGE_ERROR:
        gst_message_parse_error(message, &error, &debug_info);
        GST_ERROR("ERROR event received source: %s error %s\n", GST_OBJECT_NAME(message->src), GST_STR_NULL(error->message));
        // if the message came from a src_bin
        if (g_str_has_prefix(GST_OBJECT_NAME(message->src), "src_bin_"))
        {
            // unref the message
            gst_message_unref(message);
            // not passing the message downstream
            return GST_BUS_DROP;
        }
        g_print("ERROR from element %s: %s passing downstream\n", GST_OBJECT_NAME(message->src), error->message);
        break;
    default:
        break;
    }
    return GST_BUS_PASS;
}

void SrcBin::start_watchdog_thread()
{
    std::thread watchdog([this]()
                         { watchdog_thread(this); });
    watchdog.detach();
}

// This function is used to set the bus sync handler
void SrcBin::start_bus_sync_handler()
{
    // static bool handler_set = false;
    // if (handler_set == false)
    // {
        bus = gst_element_get_bus(bin);
        gst_bus_set_sync_handler(bus, (GstBusSyncHandler)bus_sync_handler, NULL, NULL);
        // handler_set = true;
    // }
}

// Constructor
SrcBin::SrcBin(SrcType type, const std::string &uri, bool loop_enable, gint64 max_latency, const std::string &rtsp_user, const std::string &rtsp_pass)
{
    static bool init = false;
    if (init == false)
    {
        // Initialize the app debug category
        GST_DEBUG_CATEGORY_INIT(src_bin_debug, "src_bin_debug", 2, "src_bin debug category");
        init = true;
    }
    this->name = "src_bin_" + std::to_string(bin_cnt++);
    this->id = bin_cnt;
    this->uri = uri;
    this->max_latency = max_latency;
    this->type = type;
    this->rtsp_user = rtsp_user;
    this->rtsp_pass = rtsp_pass;

    this->restarting.store(false);
    this->reset_flag = false; // used for watchdog timer
    // if the URI is file:// then enable loop
    if (uri.find("file://") != std::string::npos)
    {
        this->loop_enable = true;
    }
    // source and decodbin are created in source specific build functions
    this->source = nullptr;
    this->decodebin = nullptr;
    // create the bin
    this->bin = gst_bin_new(this->name.c_str());
    // set message-forward property
    g_object_set(G_OBJECT(bin), "message-forward", TRUE, NULL);
    // bus is available only after the bin is added to the pipeline
    this->bus = nullptr;
    // create the tee and fakesink elements
    this->tee = gst_element_factory_make("tee", ("tee" + std::to_string(id)).c_str());
    this->output_q = gst_element_factory_make("queue", ("output_q" + std::to_string(id)).c_str());
    this->fakesink_q = gst_element_factory_make("queue", ("fakesink_q" + std::to_string(id)).c_str());
    this->fakesink = gst_element_factory_make("fakesink", ("fakesink" + std::to_string(id)).c_str());
    // add the elements to the bin
    gst_bin_add_many(GST_BIN(bin), tee, output_q, fakesink_q, fakesink, NULL);
    // link the elements
    gst_element_link_many(tee, output_q, NULL);
    gst_element_link_many(tee, fakesink_q, fakesink, NULL);
    // link output_q to bin output pad
    // Get the source pad of the queue element
    GstPad *queue_src_pad = gst_element_get_static_pad(output_q, "src");

    // Create a ghost source pad and link it to the queue source pad
    GstPad *bin_src_pad = gst_ghost_pad_new("src", queue_src_pad);
    if (!gst_element_add_pad(bin, bin_src_pad))
    {
        g_warning("ERROR: Failed to add src ghost pad to bin");
    }

    // set queue properties
    ::set_queue_properties(output_q, false, 3);
    ::set_queue_properties(fakesink_q, false, 3);
    // set fakesink properties
    g_object_set(G_OBJECT(fakesink), "sync", false, "async", false, NULL);

    // set the source specific build function
    if (type == SrcType::V4L2)
    {
        if (!build_v4l2_source())
        {
            g_error("ERROR: Failed to build v4l2 source SrcBIn %d\n", id);
        }
    }
    else if (type == SrcType::URI)
    {
        if (!build_uri_source())
        {
            g_error("ERROR: Failed to build uri source SrcBIn %d\n", id);
        }
    }
    else if (type == SrcType::RTSP)
    {
        if (!build_rtsp_source())
        {
            g_error("ERROR: Failed to build rtsp source SrcBIn %d\n", id);
        }
    }
    disable_qos_in_bin(GST_BIN(bin));
}

GstElement *SrcBin::get() const { return bin; }

// callbacks

static void rtspsrc_on_pad_added(GstElement *element, GstPad *pad, gpointer data)
{
    GST_INFO("rtspsrc pad added CB\n");
    //check if this is a x-rtp pad
    GstCaps *caps = gst_pad_get_current_caps(pad);
    GstStructure *str = gst_caps_get_structure(caps, 0);
    const gchar *name = gst_structure_get_name(str);
    const gchar *media = gst_structure_get_string(str, "media");
    
    if (!g_str_has_prefix(media, "video"))
    {
        GST_INFO("rtspsrc pad ignored, not video stream, type %s\n", media);
        gst_caps_unref(caps);
        return;
    }
    GstElement *depay = static_cast<GstElement *>(data);
    GstPad *sinkpad = gst_element_get_static_pad(depay, "sink");
    GstPadLinkReturn ret = gst_pad_link(pad, sinkpad);
    if (ret != GST_PAD_LINK_OK)
    {
        g_error("ERROR: Failed to link rtspsrc pad to rtph264depay sink pad\n");
        if (ret == GST_PAD_LINK_WAS_LINKED){
            g_error("ERROR: rtph264depay sink pad was already linked, maybe 2 video streams are available?\n");
        }
    }
    gst_object_unref(sinkpad);
    // connect debug_probe_cb to pad
    // gulong probe_id = gst_pad_add_probe(pad, (GstPadProbeType)(GST_PAD_PROBE_TYPE_EVENT_BOTH), debug_probe_cb, NULL, NULL);
}
static gboolean rtspsrc_select_stream(GstElement *element, guint stream_id, GstCaps *caps, gpointer data)
{
    GST_DEBUG("rtspsrc select stream CB\n");
    GstStructure *str = gst_caps_get_structure(caps, 0);
    const gchar *name = gst_structure_get_name(str);
    const gchar *media = gst_structure_get_string(str, "media");
    // select only video streams
    if (g_str_has_prefix(media, "video"))
    {
        GST_INFO("Selecting stream %d (%s) (%s)\n", stream_id, name, media);
        return true;
    }
    GST_INFO("Stream not used %d (%s) (%s)\n", stream_id, name, media);
    return FALSE;
}

static void decodebin_on_pad_added(GstElement *decodebin, GstPad *pad, gpointer data)
{
    GST_DEBUG("Decodebin pad added CB\n");
    // check if this is a video pad
    GstCaps *caps = gst_pad_get_current_caps(pad);
    GstStructure *str = gst_caps_get_structure(caps, 0);
    const gchar *name = gst_structure_get_name(str);
    if (!g_str_has_prefix(name, "video"))
    {
        GST_INFO("Decoder pad ignored, not a video stream\n");
        gst_caps_unref(caps);
        return;
    }
    SrcBin *src_bin = static_cast<SrcBin *>(data);
    src_bin->tee_sinkpad = gst_element_get_static_pad(src_bin->tee, "sink");
    GstPadLinkReturn ret = gst_pad_link(pad, src_bin->tee_sinkpad);
    if (ret != GST_PAD_LINK_OK)
    {
        g_error("ERROR: Failed to link decodebin pad to tee sink pad\n");
    }
    src_bin->decodebin_src_pad = pad;
    // add pad probe
    //  gulong probe_id = gst_pad_add_probe(pad, (GstPadProbeType)(GST_PAD_PROBE_TYPE_EVENT_BOTH), debug_probe_cb, NULL, NULL);
    gst_caps_unref(caps);
}

gboolean SrcBin::build_uri_source()
{
    // in this pipeline the source is inside the decodebin
    decodebin = gst_element_factory_make("uridecodebin", ("decodbin_" + std::to_string(id)).c_str());
    g_object_set(G_OBJECT(decodebin), "uri", uri.c_str(), NULL);

    // Adding decodebin to the bin
    gst_bin_add(GST_BIN(bin), decodebin);

    // Adding pad-added callback to decodebin
    handler_id_decodebin_on_pad_added = g_signal_connect(decodebin, "pad-added", G_CALLBACK(decodebin_on_pad_added), this);

    if (loop_enable == true || type == SrcType::RTSP)
    {
        // add pad probe to the tee sink pad
        tee_sinkpad = gst_element_get_static_pad(tee, "sink");
        handler_id_sink_loop_pad_probe = gst_pad_add_probe(tee_sinkpad, (GstPadProbeType)(GST_PAD_PROBE_TYPE_EVENT_BOTH | GST_PAD_PROBE_TYPE_EVENT_FLUSH | GST_PAD_PROBE_TYPE_BUFFER), tee_sink_loop_pad_probe_cb, this, NULL);
    }

    return TRUE;
}

gboolean SrcBin::build_v4l2_source()
{
    source = gst_element_factory_make("v4l2src", "source");
    // add source to bin
    gst_bin_add(GST_BIN(bin), source);
    // link source to tee
    gst_element_link(source, tee);

    return TRUE;
}
// gboolean SrcBin::build_v4l2_source()
// {
//     source = gst_element_factory_make("v4l2src", "source");
//     GstCaps *mjpeg_caps = gst_caps_new_simple("image/jpeg", "width", G_TYPE_INT, 640, "height", G_TYPE_INT, 480, "framerate", GST_TYPE_FRACTION, 30, 1, NULL);
//     g_object_set(G_OBJECT(source), "caps", mjpeg_caps, NULL);
//     gst_caps_unref(mjpeg_caps);

//     decodebin = gst_element_factory_make("decodebin", "decoder");

//     gst_bin_add_many(GST_BIN(bin), source, decodebin, NULL);
//     gst_element_link_many(source, decodebin, NULL);

//     handler_id_decodebin_on_pad_added = g_signal_connect(decodebin, "pad-added", G_CALLBACK(decodebin_on_pad_added), tee);
//     return TRUE;
// }

gboolean SrcBin::build_rtspsrc_element()
{
    source = gst_element_factory_make("rtspsrc", ("source_" + std::to_string(id)).c_str());
    if (!source)
    {
        g_error("Failed to create rtspsrc element SrcBin %d.\n", id);
        return FALSE;
    }

    g_object_set(G_OBJECT(source), "location", uri.c_str(), NULL);
    g_object_set(G_OBJECT(source), "latency", max_latency, NULL);
    g_object_set(G_OBJECT(source), "user-id", rtsp_user.c_str(), NULL);
    g_object_set(G_OBJECT(source), "user-pw", rtsp_pass.c_str(), NULL);
    // g_object_set(G_OBJECT(source), "ntp-sync", true, NULL);
    // g_object_set(G_OBJECT(source), "buffer-mode", 0, NULL);
    g_object_set(G_OBJECT(source), "message-forward", true, NULL);
    g_object_set(G_OBJECT(source), "drop-on-latency", true, NULL);

    gst_bin_add(GST_BIN(bin), source);

    // rtspsrc events callbacks
    handler_id_rtspsrc_on_pad_added = g_signal_connect(source, "pad-added", G_CALLBACK(rtspsrc_on_pad_added), depay);
    handler_id_rtspsrc_select_stream = g_signal_connect(source, "select-stream", G_CALLBACK(rtspsrc_select_stream), NULL);

    return TRUE;
}

gboolean SrcBin::build_rtsp_source()
{

    depay = gst_element_factory_make("rtph264depay", ("depay_" + std::to_string(id)).c_str());
    if (!depay)
    {
        g_error("Failed to create rtph264depay element SrcBin %d.\n", id);
        return FALSE;
    }

    depay_q = gst_element_factory_make("queue", ("depay_q" + std::to_string(id)).c_str());
    if (!depay_q)
    {
        g_error("Failed to create depay queue element SrcBin %d.\n", id);
        return FALSE;
    }
    ::set_queue_properties(depay_q, false, 3);

    decodebin = gst_element_factory_make("decodebin", ("decoder_" + std::to_string(id)).c_str());
    if (!decodebin)
    {
        g_error("Failed to create decodebin element SrcBin %d.\n", id);
        return FALSE;
    }

    // gst_bin_add_many(GST_BIN(bin), source, depay, depay_q, decodebin, NULL);
    gst_bin_add_many(GST_BIN(bin), depay, depay_q, decodebin, NULL);

    if (!gst_element_link_many(depay, depay_q, decodebin, NULL))
    {
        g_error("Failed to link elements SrcBin %d.\n", id);
        return FALSE;
    }

    // decodebin events callbacks
    handler_id_decodebin_on_pad_added = g_signal_connect(decodebin, "pad-added", G_CALLBACK(decodebin_on_pad_added), this);

    build_rtspsrc_element();

    if (true) // recover from error
    {
        // add pad probe to the tee sink pad
        tee_sinkpad = gst_element_get_static_pad(tee, "sink");
        tee_sinkpad = gst_element_get_static_pad(output_q, "sink");
        handler_id_sink_loop_pad_probe = gst_pad_add_probe(tee_sinkpad, (GstPadProbeType)(GST_PAD_PROBE_TYPE_EVENT_BOTH | GST_PAD_PROBE_TYPE_EVENT_FLUSH | GST_PAD_PROBE_TYPE_BUFFER), tee_sink_loop_pad_probe_cb, this, NULL);
    }
    return TRUE;
}

SrcBin::~SrcBin()
{
    // check if the bin was not already unrefed
    // gint refcount = GST_OBJECT_REFCOUNT_VALUE(G_OBJECT(bin));
    // if (refcount > 0)
    //     gst_object_unref(bin);
}

gboolean SrcBin::set_state_playing()
{
    gboolean ret;
    ret = gst_element_set_state(bin, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE)
    {
        GST_ERROR("ERROR: Failed to set state playing\n");
        return FALSE;
    }
    return TRUE;
}

gboolean SrcBin::set_state_paused()
{
    gboolean ret;
    ret = gst_element_set_state(bin, GST_STATE_PAUSED);
    if (ret == GST_STATE_CHANGE_FAILURE)
    {
        GST_ERROR("ERROR: Failed to set state paused\n");
        return FALSE;
    }
    return TRUE;
}

gboolean SrcBin::set_state_null()
{
    gboolean ret;
    ret = gst_element_set_state(bin, GST_STATE_NULL);
    if (ret == GST_STATE_CHANGE_FAILURE)
    {
        GST_ERROR("ERROR: Failed to set state null\n");
        return FALSE;
    }
    return TRUE;
}
