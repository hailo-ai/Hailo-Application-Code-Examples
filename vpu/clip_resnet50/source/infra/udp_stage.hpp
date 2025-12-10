#pragma once

// General includes
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>
#include <tl/expected.hpp>
#include <functional>
#include <iostream>
#include <queue>
#include <thread>
#include <vector>

// Tappas includes
#include "hailo_objects.hpp"
#include "hailo_common.hpp"

// Media library includes
#include "gsthailobuffermeta.hpp"
#include "media_library/buffer_pool.hpp"
#include "media_library/media_library_types.hpp"

// Infra includes
#include "buffer.hpp"
#include "queue.hpp"
#include "stage.hpp"

// Defines
#define SRC_QUEUE_NAME "appsrc_q"

enum class EncodingType
{
    H264 = 0,
    H265,
};

class UdpModule;
using UdpModulePtr = std::shared_ptr<UdpModule>;

class UdpModule
{
private:
    std::string m_name;
    std::string m_host;
    std::string m_port;
    EncodingType m_type;

    GstAppSrc *m_appsrc;
    GstCaps *m_appsrc_caps;
    GMainLoop *m_main_loop;
    std::shared_ptr<std::thread> m_main_loop_thread;
    GstElement *m_pipeline;
    
public:
    static tl::expected<UdpModulePtr, AppStatus> create(std::string name, std::string host, std::string port, EncodingType type);

    ~UdpModule();
    UdpModule(std::string name, std::string host, std::string port, EncodingType type, AppStatus &status);

public:
    AppStatus start();
    AppStatus stop();
    AppStatus add_buffer(HailoMediaLibraryBufferPtr ptr, size_t size);

    /**
     * Below are public functions that are not part of the public API
     * but are public for gstreamer callbacks.
     */
public:
    void on_fps_measurement(GstElement *fpssink, gdouble fps, gdouble droprate,
                            gdouble avgfps);
    gboolean on_bus_call(GstBus *bus, GstMessage *msg);

private:
    static void fps_measurement(GstElement *fpssink, gdouble fps,
                                gdouble droprate, gdouble avgfps,
                                gpointer user_data)
    {
        UdpModule *udp_module = static_cast<UdpModule *>(user_data);
        udp_module->on_fps_measurement(fpssink, fps, droprate, avgfps);
    }
    static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer user_data)
    {
        UdpModule *udp_module = static_cast<UdpModule *>(user_data);
        return udp_module->on_bus_call(bus, msg);
    }

private:
    void set_gst_callbacks();
    GstFlowReturn add_buffer_internal(GstBuffer *buffer);
    std::string create_pipeline_string();
};

tl::expected<UdpModulePtr, AppStatus>
UdpModule::create(std::string name, std::string host, std::string port, EncodingType type)
{
    AppStatus status = AppStatus::UNINITIALIZED;
    UdpModulePtr udp_module = std::make_shared<UdpModule>(name, host, port, type, status);
    if (status != AppStatus::SUCCESS)
    {
        return tl::make_unexpected(status);
    }
    return udp_module;
}

UdpModule::UdpModule(std::string name, std::string host, std::string port, EncodingType type, AppStatus &status)
    : m_name(name), m_host(host), m_port(port), m_type(type)
{
    // Initialize gstreamer
    gst_init(nullptr, nullptr);
    m_pipeline = gst_parse_launch(create_pipeline_string().c_str(), NULL);
    if (!m_pipeline)
    {
        std::cerr << "Failed create UDP pipeline" << std::endl;
        status = AppStatus::CONFIGURATION_ERROR;
        return;
    }
    gst_bus_add_watch(gst_element_get_bus(m_pipeline), (GstBusFunc)bus_call, this);
    m_main_loop = g_main_loop_new(NULL, FALSE);
    this->set_gst_callbacks();

    status = AppStatus::SUCCESS;
}

UdpModule::~UdpModule()
{
    gst_caps_unref(m_appsrc_caps);
    gst_object_unref(m_pipeline);
}

AppStatus UdpModule::start()
{
    GstStateChangeReturn ret =
        gst_element_set_state(m_pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE)
    {
        std::cerr << "Failed to start UDP pipeline" << std::endl;
        return AppStatus::PIPELINE_ERROR;
    }
    m_main_loop_thread = std::make_shared<std::thread>(
        [this]()
        { g_main_loop_run(m_main_loop); });

    return AppStatus::SUCCESS;
}

AppStatus UdpModule::stop()
{
    gboolean ret = gst_element_send_event(m_pipeline, gst_event_new_eos());
    if (!ret)
    {
        std::cerr << "Failed to stop the UDP pipeline" << std::endl;
        return AppStatus::PIPELINE_ERROR;
    }

    gst_element_set_state(m_pipeline, GST_STATE_NULL);
    g_main_loop_quit(m_main_loop);
    m_main_loop_thread->join();
    return AppStatus::SUCCESS;
}

/**
 * Create the gstreamer pipeline as string
 *
 * @return A string containing the gstreamer pipeline.
 */
std::string UdpModule::create_pipeline_string()
{
    std::string pipeline = "";

    std::string caps_type;
    std::string rtp_payloader;
    std::string encoding_name;
    if (m_type == EncodingType::H264)
    {
        caps_type = "video/x-h264";
        rtp_payloader = "rtph264pay";
        encoding_name = "H264";
    }
    else
    {
        caps_type = "video/x-h265";
        rtp_payloader = "rtph264pay";
        encoding_name = "H265";
    }

    std::ostringstream caps2;
    caps2 << caps_type << ",framerate=30/1,stream-format=byte-stream,alignment=au";

    std::ostringstream udp_sink;
    udp_sink << "udpsink host=" << m_host << " port=" << m_port;

    pipeline =
        "appsrc do-timestamp=true format=time block=true is-live=true max-bytes=0 "
        "max-buffers=1 name=udp_src ! "
        "queue name=" +
        std::string(SRC_QUEUE_NAME) + " leaky=no max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! " +
        caps2.str() + " ! " +
        "tee name=udp_tee "
        "udp_tee. ! "
            "queue leaky=no max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! " +
            rtp_payloader + " ! application/x-rtp, media=video, encoding-name=" + encoding_name + " ! " +
            udp_sink.str() + " name=udp_sink sync=true "
        "udp_tee. ! "
            "queue leaky=no max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! "
            "fpsdisplaysink fps-update-interval=2000 signal-fps-measurements=true name=fpsdisplaysink "
            "text-overlay=false sync=true video-sink=fakesink ";

    std::cout << "Pipeline: " << pipeline << std::endl;

    return pipeline;
}

// /**
//  * Print the FPS of the pipeline
//  *
//  * @note Prints the FPS to the stdout.
//  */
void UdpModule::on_fps_measurement(GstElement *fpsdisplaysink,
                                    gdouble fps,
                                    gdouble droprate,
                                    gdouble avgfps)
{
    gchar *name;
    g_object_get(G_OBJECT(fpsdisplaysink), "name", &name, NULL);
    std::cout << m_name << ", DROP RATE: " << droprate << " FPS: " << fps << " AVG_FPS: " << avgfps << std::endl;
    g_free(name);
}

/**
 * Set the callbacks
 *
 * @param[in] pipeline        The pipeline as a GstElement.
 */
void UdpModule::set_gst_callbacks()
{
    // set fps callbacks
    GstElement *fpssink = gst_bin_get_by_name(GST_BIN(m_pipeline), "fpsdisplaysink");
    if (1)
    {
        g_signal_connect(fpssink, "fps-measurements", G_CALLBACK(fps_measurement),
                         this);
    }
    gst_object_unref(fpssink);

    GstElement *appsrc = gst_bin_get_by_name(GST_BIN(m_pipeline), "udp_src");
    m_appsrc = GST_APP_SRC(appsrc);
    gst_object_unref(appsrc);
}

gboolean UdpModule::on_bus_call(GstBus *bus, GstMessage *msg)
{
    switch (GST_MESSAGE_TYPE(msg))
    {
    case GST_MESSAGE_EOS:
    {
        //TODO: EOS never received
        gst_element_set_state(m_pipeline, GST_STATE_NULL);
        g_main_loop_quit(m_main_loop);
        m_main_loop_thread->join();
        break;
    }
    case GST_MESSAGE_ERROR:
    {
        gchar *debug;
        GError *err;

        gst_message_parse_error(msg, &err, &debug);
        std::cerr << "Received an error message from the UDP pipeline: " << err->message << std::endl;
        g_error_free(err);
        g_free(debug);
        gst_element_set_state(m_pipeline, GST_STATE_NULL);
        g_main_loop_quit(m_main_loop);
        m_main_loop_thread->join();
        break;
    }
    default:
        break;
    }
    return TRUE;
}

struct UdpPtrWrapper
{
    HailoMediaLibraryBufferPtr ptr;
};

static inline void hailo_media_library_udp_release(UdpPtrWrapper *wrapper)
{
    delete wrapper;
}

AppStatus UdpModule::add_buffer(HailoMediaLibraryBufferPtr ptr, size_t size)
{
    // convert to GstBuffer
    UdpPtrWrapper *wrapper = new UdpPtrWrapper();
    wrapper->ptr = ptr;
    GstBuffer *gst_buffer = gst_buffer_new_wrapped_full(GST_MEMORY_FLAG_PHYSICALLY_CONTIGUOUS,
                                                        ptr->get_plane_ptr(0),
                                                        ptr->get_plane_size(0),
                                                        0, size, wrapper, GDestroyNotify(hailo_media_library_udp_release));
    gst_buffer_add_hailo_buffer_meta(gst_buffer, ptr, size);
    
    GstFlowReturn ret = this->add_buffer_internal(gst_buffer);
    if (ret != GST_FLOW_OK)
    {
        return AppStatus::PIPELINE_ERROR;
    }
    return AppStatus::SUCCESS;
}

GstFlowReturn UdpModule::add_buffer_internal(GstBuffer *buffer)
{
    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(m_appsrc), buffer);
    if (ret != GST_FLOW_OK)
    {
        GST_ERROR_OBJECT(m_appsrc, "Failed to push buffer to appsrc");
        return ret;
    }
    return ret;
}
