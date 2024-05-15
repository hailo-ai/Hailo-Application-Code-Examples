#include <iostream>
#include <regex>
#include <glib.h>
#include <string>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <osd.hpp>


void printBashPipeline(const std::string& pipeline, bool add_launcher = false) {
    // Define regex patterns
    std::regex pat_parentheses("!\\s+([^!]+[()][^!]+)\\s+!");
    std::regex pat_spaces("\\s+");

    // Surround elements with parentheses with single quotes
    std::string bash_pipeline_str = std::regex_replace(pipeline, pat_parentheses, "! '$1' !");

    // Add launcher and remove duplicate spaces
    bash_pipeline_str = std::regex_replace(bash_pipeline_str, pat_spaces, " ");
    if (add_launcher)
        bash_pipeline_str = "gst-launch-1.0 -v " + bash_pipeline_str;

    // Print the formatted bash pipeline
    std::cout << "\nBash Pipeline:" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << bash_pipeline_str << std::endl;
    std::cout << "------------------------------------------\n" << std::endl;
}

int main(int argc, char *argv[]) {

    gchar *currentDir = g_get_current_dir();
    const std::string basePath = currentDir;
    g_free(currentDir);
    const std::string RES_DIR = basePath + "/resources";
    const std::string CONF_DIR = RES_DIR + "/configs";

    GstFlowReturn ret;
    gst_init(&argc, &argv);

    const std::string QUEUE   = " ! queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ";
    const std::string TRACKER = " ! hailotracker name=hailo_tracker keep-past-metadata=true kalman-dist-thr=.5 iou-thr=.6 keep-tracked-frames=2 keep-lost-frames=2 ";
    const std::string OVERLAY = " ! hailooverlay line-thickness=5 font-thickness=3 qos=false ";
    const std::string ENCODER = " ! hailoencoder enforce-caps=false config-file-path="+CONF_DIR+"/enc_osd.json ! " \
                                "   h264parse config-interval=-1 ! video/x-h264,framerate=30/1 ";
    const std::string RTP     = " ! rtph264pay ! application/x-rtp, media=(string)video, encoding-name=(string)H264 ! udpsink host=10.0.0.2 port=5002 "
                                "  name=udp_sink2 sync=false ";
    const std::string CROPPER = " ! hailocropper so-path=/usr/lib/hailo-post-processes/cropping_algorithms/libmspn.so function-name=create_crops_only_person " \
                                " internal-offset=true name=cropper hailoaggregator name=agg ";
    const std::string OSD =     " ! hailoosd name=osd config-file-path="+CONF_DIR+"/enc_osd.json ";

    //TODO: Implement //////////////
    const std::string TEMPERING = QUEUE;
    const std::string LOITERING = "";
    const std::string FIRE_CLASS = "";
    ////////////////////////////////

    const std::string FIRE_DET_HAILONET = " ! hailonet hef-path="+RES_DIR+"/yolov8s_vga_nv12.hef scheduling-algorithm=1 scheduler-threshold=5 scheduler-timeout-ms=100 vdevice-group-id=1 vdevice-key=1 outputs-max-pool-size=4 outputs-min-pool-size=2 ";
    const std::string FIRE_DET_PP = " ! hailofilter function-name=yolov8_no_persons so-path="+RES_DIR+"/libyolo_hailortpp.so qos=false ";
    const std::string FIRE_DET = QUEUE + FIRE_DET_HAILONET + QUEUE + FIRE_DET_PP; 

    const std::string OD_PIPELINE = QUEUE + \
                            " ! hailonet hef-path="+RES_DIR+"/yolov5s_hd_nv12.hef scheduling-algorithm=1" \
                            "   scheduler-threshold=5 scheduler-timeout-ms=100 vdevice-group-id=1 vdevice-key=1 outputs-max-pool-size=4" \
                            "   outputs-min-pool-size=2 ! video/x-raw,framerate=30/1 "+ QUEUE +\
                            " ! hailofilter config-path="+CONF_DIR+"/yolov5.json " \
                            "   so-path="+RES_DIR+"/libyolo_hailortpp.so qos=false function-name=yolov5s_only_persons name=detection ";
    const std::string POSE_EST_PIPELINE = " ! hailonet hef-path="+RES_DIR+"/mspn_regnetx_800mf_nv12_hailo15h.hef vdevice-group-id=1" \
                                    " vdevice-key=1 outputs-max-pool-size=4 outputs-min-pool-size=2 ! video/x-raw, framerate=30/1 "+ QUEUE+ \
                                    " ! hailofilter name=pose-estimation so-path=/usr/lib/hailo-post-processes/libmspn_post.so qos=false ";

    const std::string FRONTEND_PIPELINE = "v4l2src device=/dev/video0 io-mode=mmap !" \
                                          " video/x-raw,format=NV12,width=3840,height=2160,framerate=30/1 !" \
                                          " queue leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 !" \
                                          " hailofrontend config-file-path="+CONF_DIR+"/frontend_config_3_streams.json name=frontend hailomuxer name=hmux1 hailomuxer name=hmux2 ";
    const std::string VGA_PIPELINE  = QUEUE + FIRE_DET + FIRE_CLASS + " ! hmux1.";

    const std::string HD_PIPELINE = OD_PIPELINE + QUEUE + TRACKER + QUEUE + CROPPER + \
                                            "cropper." + TEMPERING + LOITERING + " ! agg. " +\
                                            "cropper." + QUEUE + POSE_EST_PIPELINE + QUEUE + " ! agg. " +\
                                        "agg. " + QUEUE + " ! hmux1. hmux1. " + QUEUE + " ! hmux2. ";
    const std::string FHD_PIPELINE = QUEUE + "! hmux2. hmux2. " + QUEUE + OVERLAY + OSD + QUEUE + ENCODER + " ! tee name=udp_tee "\
                                           + " udp_tee."  + QUEUE + " ! fpsdisplaysink video-sink=fakesink name=hailo_display sync=false text-overlay=false"\
                                           + " udp_tee. " + QUEUE + RTP;
   
    const std::string FULL_PIPELINE = FRONTEND_PIPELINE + \
                                            "frontend." + FHD_PIPELINE + \
                                            "frontend." + HD_PIPELINE + \
                                            "frontend." + VGA_PIPELINE;

    if (FULL_PIPELINE.find('\'') != std::string::npos) {
        std::cerr << "Error: Pipeline contains a single quote character (')." << std::endl;
        return 1;
    }
    
    // Print bash-compatible string for debug
    printBashPipeline(FULL_PIPELINE, true);

    const gchar* FULL_PIPELINE_gstr = FULL_PIPELINE.c_str();

    // Create the pipeline
    GstElement *pipeline = gst_parse_launch(FULL_PIPELINE_gstr, NULL);

    // Start playing the pipeline
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    
    int frame_counter = 0;  // Counter to keep track of frames processed
    std::shared_ptr<osd::Blender> blender;
    GstElement *hailoosd = gst_bin_get_by_name(GST_BIN(pipeline), "osd");
    GValue val = G_VALUE_INIT;
    g_object_get_property(G_OBJECT(hailoosd), "blender", &val);
    void *value_ptr = g_value_get_pointer(&val);
    blender = reinterpret_cast<osd::Blender *>(value_ptr)->shared_from_this();
    g_value_unset(&val);
    gst_object_unref(hailoosd);
    std::string FONTS_DIR = "/usr/share/fonts/ttf/";

    // // Wait until error or EOS
    GstBus *bus = gst_element_get_bus(pipeline);
    GstMessage *msg = nullptr;
    
    while (true) {
        // Poll messages from the bus
        msg = gst_bus_poll(bus, GST_MESSAGE_ANY, GST_MSECOND); // Poll for 1 milisecond
        if (msg != nullptr) {
            // Check if the message is an EOS or an error
            if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_EOS || GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                gst_message_unref(msg);
                break;
            }
            gst_message_unref(msg);
        }
    
        // Increment the frame counter
        frame_counter++;        

        osd::TextOverlay new_text = osd::TextOverlay(
            "id_example"+std::to_string(frame_counter), // Text overlay ID
            0.04,                                        // X position
            0.1,                                        // Y position
            "Cam",                                      // Text content
            osd::rgb_color_t{255, 155, 0},              // Text color (RGB: 0, 0, 255) - blue
            osd::rgb_color_t{255, 255, 255},            // Background color (RGB: 255, 255, 255) - white
            40.0,                                       // Font size
            1,                                          // Border thickness
            1,                                          // Border opacity
            FONTS_DIR+"LiberationMono-Regular.ttf",     // Font path
            frame_counter*2,                            // Text rotation angle
            osd::CENTER                                 // Text alignment
        );        
        blender->add_overlay_async(new_text);
        if (frame_counter>1)
            blender->remove_overlay("id_example"+std::to_string(frame_counter-1));      

    }

    
    // Clean up
    gst_object_unref(bus);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    return 0;
}
