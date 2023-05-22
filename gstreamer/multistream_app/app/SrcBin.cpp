#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <gst/gst.h>
#include <string>

#include "SrcBin.hpp"

GST_DEBUG_CATEGORY (src_bin_debug);
#define GST_CAT_DEFAULT src_bin_debug

guint SrcBin::bin_cnt = 0;

// Declare a static pad template for the source pad of the bin
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);
#include <gst/gst.h>

void disable_qos_in_bin(GstBin *bin) {
    GstIterator *it;
    GValue item = G_VALUE_INIT;
    gboolean done = FALSE;

    g_return_if_fail(GST_IS_BIN(bin));

    it = gst_bin_iterate_elements(bin);
    while (!done) {
        switch (gst_iterator_next(it, &item)) {
        case GST_ITERATOR_OK: {
            GstElement *element = GST_ELEMENT(g_value_get_object(&item));

            if (g_object_class_find_property(G_OBJECT_GET_CLASS(element), "qos")) {
                g_object_set(G_OBJECT(element), "qos", FALSE, NULL);
            }

            g_value_reset(&item);
            break;
        }

        case GST_ITERATOR_RESYNC:
            gst_iterator_resync(it);
            break;

        case GST_ITERATOR_ERROR:
        case GST_ITERATOR_DONE:
            done = TRUE;
            break;
        }
    }
    g_value_unset(&item);
    gst_iterator_free(it);
}

// a function to set queue properties
static void set_queue_properties(GstElement *queue, gboolean leaky = false, guint max_size_buffers = 3)
{
    g_object_set(G_OBJECT(queue), "leaky", leaky, NULL);
    g_object_set(G_OBJECT(queue), "max-size-buffers", max_size_buffers, NULL);
    g_object_set(G_OBJECT(queue), "max-size-bytes", 0, NULL);
    g_object_set(G_OBJECT(queue), "max-size-time", 0, NULL);
}

// Probe callback debug events

static GstPadProbeReturn
debug_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  GstEvent *event = GST_PAD_PROBE_INFO_EVENT(info);
  const gchar *event_name = GST_EVENT_TYPE_NAME(event);
  GstPadDirection direction = gst_pad_get_direction(pad);
  const gchar *direction_name = (direction == GST_PAD_SRC) ? "src" : "sink";
  GstElement *parent = gst_pad_get_parent_element(pad);

  GST_DEBUG("Event '%s' on %s pad '%s', parent element '%s'\n",
    event_name,
    direction_name,
    GST_PAD_NAME(pad),
    GST_ELEMENT_NAME(parent));

  gst_object_unref(parent);

  return GST_PAD_PROBE_OK;
}

// define seek_to_start() function
static gboolean seek_to_start(gpointer data)
{
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

static gboolean restart_source(gpointer data)
{
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
    // // for each element in the bin make sure that the state is NULL
    // GstIterator *it = gst_bin_iterate_elements(GST_BIN(src_bin->bin));
    // GValue item = G_VALUE_INIT;
    // while (gst_iterator_next(it, &item) == GST_ITERATOR_OK)
    // {
    //     GstElement *element = static_cast<GstElement *>(g_value_get_object(&item));
    //     gst_element_set_state(element, GST_STATE_NULL);
    //     gst_element_get_state(element, NULL, NULL, GST_CLOCK_TIME_NONE);
    //     g_debug("Element %s state set to NULL\n", GST_OBJECT_NAME(element));
    // }
    GST_WARNING("Restarting source...\n");
    //src_bin->rebuild_rtsp_source();
    
    //wait for 1 sec
    // GST_INFO("Waiting for 1 sec\n");
    // g_usleep(1000000);
    // resume the pipeline
    gst_element_set_state(src_bin->bin, GST_STATE_PLAYING);
    // wait for state change to complete
    ret = gst_element_get_state(src_bin->bin, NULL, NULL, GST_CLOCK_TIME_NONE);
    if (ret == GST_STATE_CHANGE_FAILURE)
    {
        GST_ERROR("ERROR: Failed to set state to PLAYING\n");
        return true;
    }
    GST_INFO("Source restarted\n");
    src_bin->restarting.store(false);
    return false;
}
// Probe callback to enable loop on the source
static GstPadProbeReturn tee_sink_loop_pad_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer data)
{
    GstEvent *event = GST_PAD_PROBE_INFO_EVENT(info);
    SrcBin *src_bin = static_cast<SrcBin *>(data);
    static int cnt = 0;
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
            cnt++;
            GST_INFO("cnt: %d\n", cnt);
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

void watchdog_thread(gpointer data, uint32_t timeout=5)
{
    SrcBin *src_bin = static_cast<SrcBin *>(data);
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(timeout));
        if (src_bin->restarting.load() == true) {
            continue;
        }
        // If the flag hasn't been reset, we assume the stream has stalled
        if (src_bin->reset_flag.exchange(false) == false) {
            GST_WARNING("Stream stalled, restarting source\n");
            src_bin->restarting.store(true);
            // schedule a pipeline restart
            g_timeout_add(2000, (GSourceFunc)restart_source, src_bin);
        }
    }
}

static GstBusSyncReply bus_sync_handler(GstBus *bus, GstMessage *message, gpointer data)
{
    SrcBin *src_bin = static_cast<SrcBin *>(data);
    const GstStructure *structure;
    GError *err = NULL;
    gchar *debuGST_INFO = NULL;
    // print message type and source in one line
    // GST_INFO("Message source: %s, type: %s\n", GST_OBJECT_NAME(message->src), GST_MESSAGE_TYPE_NAME(message));
    switch (GST_MESSAGE_TYPE(message))
    {
    case GST_MESSAGE_EOS:
        GST_INFO("EOS event received source: %s\n", GST_OBJECT_NAME(message->src));
        break;
    case GST_MESSAGE_ERROR:
        gst_message_parse_error(message, &err, &debuGST_INFO);
        GST_ERROR("ERROR event received source: %s error %s\n", GST_OBJECT_NAME(message->src), GST_STR_NULL(err->message));
        break;
    case GST_MESSAGE_ELEMENT:
        // get structure from the message
        structure = gst_message_get_structure(message);
        // print structure name
        GST_INFO("Structure name: %s\n", gst_structure_get_name(structure));
        // if this is a GstBinForwarded message get the childs message
        if (gst_structure_has_name(structure, "GstBinForwarded"))
        {
            GstMessage *child_msg;
            gst_structure_get(structure, "message", GST_TYPE_MESSAGE, &child_msg, NULL);
            // print child message type and source
            GST_INFO("Child message source: %s, type: %s\n", GST_OBJECT_NAME(child_msg->src), GST_MESSAGE_TYPE_NAME(child_msg));
            // if this is an EOS message
            if (GST_MESSAGE_TYPE(child_msg) == GST_MESSAGE_EOS)
            {
                GST_INFO("EOS event received source: %s\n", GST_OBJECT_NAME(child_msg->src));
            }
            // if this is an ERROR message
            if (GST_MESSAGE_TYPE(child_msg) == GST_MESSAGE_ERROR)
            {
                gst_message_parse_error(child_msg, &err, &debuGST_INFO);
                GST_ERROR("ERROR event received source: %s error %s\n", GST_OBJECT_NAME(child_msg->src), GST_STR_NULL(err->message));
            }
            // if this is a GST_MESSAGE_ASYNC_DONE set play state
            if (GST_MESSAGE_TYPE(child_msg) == GST_MESSAGE_ASYNC_DONE)
            {
                GST_INFO("ASYNC_DONE event received source: %s\n", GST_OBJECT_NAME(child_msg->src));
                // set the state to playing
                src_bin->set_state_playing();
                src_bin->restarting.store(false);
            }
            gst_message_unref(child_msg);
        }
        break;
    default:
        break;
    }
    // TBD
    return GST_BUS_PASS;

    // // unref the message
    // gst_message_unref(message);
    // // not passing the message downstream
    // return GST_BUS_DROP;
}

// Constructor
SrcBin::SrcBin(SrcType type, const std::string &uri, bool live_src, bool loop_enable, gint64 max_latency)
{
    // Initialize the app debug category
    GST_DEBUG_CATEGORY_INIT (src_bin_debug, "src_bin_debug", 2, "src_bin debug category");
    gst_debug_set_threshold_for_name("src_bin_debug", GST_LEVEL_WARNING);
    this->name = "src_bin_" + std::to_string(bin_cnt++);
    this->id = bin_cnt;
    this->uri = uri;
    this->live_src = live_src;
    this->max_latency = max_latency;
    this->type = type;
  
    this->restarting.store(false); // TBD when the pipeline starts, the source is restarting
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
    std::thread watchdog([this]() { watchdog_thread(this); });
    watchdog.detach();
}

gboolean SrcBin::set_bus_handler()
{
    // get the bus
    bus = gst_element_get_bus(bin);
    if (!bus)
    {
        GST_INFO("ERROR: Failed to get bus\n");
        return FALSE;
    }
    // add bus callback for errors
    //  g_signal_connect(bus, "message::error", G_CALLBACK(bus_error_cb), NULL);
    gst_bus_set_sync_handler(bus, bus_sync_handler, this, NULL);
    gst_object_unref(bus);
    return TRUE;
}

GstElement *SrcBin::get() const { return bin; }

// callbacks

static void rtspsrc_on_pad_added(GstElement *element, GstPad *pad, gpointer data)
{
    GST_INFO("rtspsrc pad added CB\n");
    // check if this is a x-rtp pad
    // GstCaps *caps = gst_pad_get_current_caps(pad);
    // GstStructure *str = gst_caps_get_structure(caps, 0);
    // const gchar *name = gst_structure_get_name(str);
    // if (!g_str_has_prefix(name, "application/x-rtp"))
    // {
    //     GST_INFO("rtspsrc pad ignored, not a x-rtp stream\n");
    //     GST_INFO("rtspsrc pad ignored, not a x-rtp stream %s\n", name);
    //     gst_caps_unref(caps);
    //     return;
    // }
    GstElement *depay = static_cast<GstElement *>(data);
    GstPad *sinkpad = gst_element_get_static_pad(depay, "sink");
    GstPadLinkReturn ret = gst_pad_link(pad, sinkpad);
    if (ret != GST_PAD_LINK_OK)
    {
        g_error("ERROR: Failed to link rtspsrc pad to rtph264depay sink pad\n");
    }
    gst_object_unref(sinkpad);
    //connect debug_probe_cb to pad
    //gulong probe_id = gst_pad_add_probe(pad, (GstPadProbeType)(GST_PAD_PROBE_TYPE_EVENT_BOTH), debug_probe_cb, NULL, NULL);
}
static gboolean rtspsrc_select_stream(GstElement *element, guint stream_id, GstCaps *caps, gpointer data)
{
    GST_INFO("rtspsrc select stream CB\n");
    GstStructure *str = gst_caps_get_structure(caps, 0);
    const gchar *name = gst_structure_get_name(str);
    const gchar *media = gst_structure_get_string(str, "media");
    GST_INFO("Selecting stream %d (%s) (%s)\n", stream_id, name, media);
    // select only video streams
    if (g_str_has_prefix(name, "video"))
    {
        return true;
    }
    return false;
}

static void decodebin_on_pad_added(GstElement *decodebin, GstPad *pad, gpointer data)
{
    GST_INFO("Decodebin pad added CB\n");
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
    //add pad probe
    // gulong probe_id = gst_pad_add_probe(pad, (GstPadProbeType)(GST_PAD_PROBE_TYPE_EVENT_BOTH), debug_probe_cb, NULL, NULL);
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
        // GST_INFOerr("ERROR: Failed to create rtspsrc element.\n");
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
    // g_object_set(G_OBJECT(source), "drop-on-latency", true, NULL);
    

    gst_bin_add(GST_BIN(bin), source);

    // rtspsrc events callbacks
    handler_id_rtspsrc_on_pad_added = g_signal_connect(source, "pad-added", G_CALLBACK(rtspsrc_on_pad_added), depay);
    handler_id_rtspsrc_select_stream = g_signal_connect(source, "select-stream", G_CALLBACK(rtspsrc_select_stream), NULL);

    return TRUE;
}
gboolean SrcBin::rebuild_rtsp_source(){
    // remove rtspsrc element
    gst_bin_remove(GST_BIN(bin), source);
    gst_element_set_state(source, GST_STATE_NULL);
    // unref all callbacks
    g_signal_handler_disconnect(source, handler_id_rtspsrc_on_pad_added);
    g_signal_handler_disconnect(source, handler_id_rtspsrc_select_stream);
    // unref rtspsrc element
    gst_object_unref(source);
    
    // remove depay from bin
    gst_bin_remove(GST_BIN(bin), depay);
    // unref depay
    gst_object_unref(depay);
    // remove depay_q from bin
    gst_bin_remove(GST_BIN(bin), depay_q);
    // unref depay_q
    gst_object_unref(depay_q);

    // remove decodebin from bin
    // remove callbacks
    g_signal_handler_disconnect(decodebin, handler_id_decodebin_on_pad_added);
    // unlink decodebin from tee
    gst_element_unlink(decodebin, tee);
    // remove decodebin from bin
    gst_bin_remove(GST_BIN(bin), decodebin);
    // unref decodebin
    gst_object_unref(decodebin);

    //remove pad probe
    gst_pad_remove_probe(tee_sinkpad, handler_id_sink_loop_pad_probe);
    // build rtsp source
    return build_rtsp_source();
}
// gboolean SrcBin::rebuild_rtspsrc_element()
// {
//     // remove old rtspsrc element
//     gst_bin_remove(GST_BIN(bin), source);
//     gst_element_set_state(source, GST_STATE_NULL);
//     // unref all callbacks
//     g_signal_handler_disconnect(source, handler_id_rtspsrc_on_pad_added);
//     g_signal_handler_disconnect(source, handler_id_rtspsrc_select_stream);
//     // unref rtspsrc element
//     gst_object_unref(source);
//     // build new rtspsrc element
//     return build_rtspsrc_element();
// }
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
