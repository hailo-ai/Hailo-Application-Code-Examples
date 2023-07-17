#pragma once
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <gst/gst.h>
#include <string>

class SrcBin {
public:
    enum class SrcType {
        V4L2,
        URI,
        RTSP
    };

    SrcType type = SrcType::URI;
    std::string uri = "";
    std::string rtsp_user = "";
    std::string rtsp_pass = "";
    bool loop_enable = false;
    gint64 max_latency = 33;

    static guint bin_cnt;
    guint id;
    guint64 current_base_timestamp = 0;
    guint64 next_base_timestamp = 0;
    
    std::string name;
    GstElement *bin;
    GstElement *source;
    GstElement *depay;
    GstElement *depay_q;
    GstElement *decodebin;
    GstElement *tee;
    GstElement *output_q;
    GstElement *fakesink_q;
    GstElement *fakesink;
    GstPad *src_pad;
    GstPad *decodebin_src_pad;
    GstPad *tee_sinkpad;
    GstBus *bus;

    SrcBin(SrcType type = SrcType::URI, const std::string& uri = "", bool loop_enable = false, gint64 max_latency = 0, const std::string& rtsp_user = "root", const std::string& rtsp_pass = "hailo");

    ~SrcBin();
    GstElement* get() const;
    gboolean set_state_playing();
    gboolean set_state_paused();
    gboolean set_state_null();
    gboolean rebuild_rtsp_source();
    void start_watchdog_thread();
    void start_bus_sync_handler();
    std::atomic<bool> restarting; //used for to indicate that the source is restarting
    std::atomic<bool> reset_flag; //used for watchdog timer 

private:
    gboolean build_bin();
    gboolean build_uri_source();
    gboolean build_v4l2_source();
    gboolean build_rtsp_source();
    gboolean build_rtspsrc_element();

    //handlers pointers
    gulong handler_id_rtspsrc_on_pad_added;
    gulong handler_id_rtspsrc_select_stream;
    gulong handler_id_decodebin_on_pad_added;
    gulong handler_id_sink_loop_pad_probe;
};