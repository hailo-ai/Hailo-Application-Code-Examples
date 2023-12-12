#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/app/gstappsink.h>
#include <glib.h>
#include <iostream>
#include <fstream>
#include <thread>
#include <iostream>
#include <fstream>
#include <sstream>
#include <tl/expected.hpp>
#include "media_library/vision_pre_proc.hpp"
#include "buffer_utils.hpp"
#include "v4l2_vsm/hailo_vsm_meta.h"
#include "v4l2_vsm/hailo_vsm.h"
#include "media_library/encoder.hpp"
#include "media_library/yolov8.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>
#include <gst/app/gstappsrc.h>


const char* VISION_PREPROC_CONFIG_FILE = "/usr/bin/preproc_config_example.json";
const char* ENCODER_OSD_CONFIG_FILE = "/usr/bin/encoder_config_example.json";

static gboolean waiting_eos = FALSE;
static gboolean caught_sigint = FALSE;
struct MediaLibrary
{
  MediaLibraryVisionPreProcPtr vision_preproc;
  MediaLibraryEncoderPtr encoder;
  Yolov8* yolov8;
  std::future<void> encoder_thread;
  std::atomic<bool> stopSignal;
  std::queue<std::pair<char*, uint32_t>> queue;
  std::mutex mut;
  
  void yolov8_init(){
    int width = vision_preproc->get_output_video_config().resolutions[0].dimensions.destination_width;
    int height = vision_preproc->get_output_video_config().resolutions[0].dimensions.destination_height;
    
    yolov8 = new Yolov8(std::string("yolov8n.hef"), width, height);
  }

  ~MediaLibrary(){
    delete yolov8;
  }

  // Stops the encoder and the yolov8
  void stop(){
    encoder->stop(); 
    stopSignal = true;
    yolov8->stop();
  }

  // Wait for queue to fill, and add the frames to the encoder
  // The queue contains buffers with detections
  void run_encoder_async(boost::lockfree::queue<hailo_media_library_buffer*> *queue){
    while(!stopSignal){
      while (queue->empty() && !stopSignal)
      {
          sleep(0.01);
      }
      if(stopSignal){
        return;
      }
      hailo_media_library_buffer* buffer;
      queue->pop(buffer);
      encoder->add_buffer(std::make_shared<hailo_media_library_buffer>(std::move(*buffer)));
      // Delete the allocation from line 252
      delete buffer;
    }
  }

};

static void sigint_restore(void)
{
  struct sigaction action;

  memset (&action, 0, sizeof (action));
  action.sa_handler = SIG_DFL;

  sigaction (SIGINT, &action, NULL);
}

/* we only use sighandler here because the registers are not important */
static void
sigint_handler_sighandler(int signum)
{
  /* If we were waiting for an EOS, we still want to catch
   * the next signal to shutdown properly (and the following one
   * will quit the program). */
  if (waiting_eos) {
    waiting_eos = FALSE;
  } else {
    sigint_restore();
  }
  /* we set a flag that is checked by the mainloop, we cannot do much in the
   * interrupt handler (no mutex or other blocking stuff) */
  caught_sigint = TRUE;
}

void add_sigint_handler(void)
{
  struct sigaction action;

  memset (&action, 0, sizeof (action));
  action.sa_handler = sigint_handler_sighandler;

  sigaction (SIGINT, &action, NULL);
}

/* is called every 250 milliseconds (4 times a second), the interrupt handler
 * will set a flag for us. We react to this by posting a message. */
static gboolean check_sigint(GstElement * pipeline)
{
  if (!caught_sigint) {
    return TRUE;
  } else {
    caught_sigint = FALSE;
    waiting_eos = TRUE;
    GST_INFO_OBJECT(pipeline, "handling interrupt. send EOS");
    GST_ERROR_OBJECT(pipeline, "handling interrupt. send EOS");
    gst_element_send_event(pipeline, gst_event_new_eos());

    /* remove timeout handler */
    return FALSE;
  }
}

GstFlowReturn wait_for_end_of_pipeline(GstElement *pipeline)
{
    GstBus *bus;
    GstMessage *msg;
    GstFlowReturn ret = GST_FLOW_ERROR;
    bus = gst_element_get_bus(pipeline);
    gboolean done = FALSE;
    // This function blocks until an error or EOS message is received.
    while(!done)
    {
        msg = gst_bus_timed_pop_filtered(bus, GST_MSECOND * 250, (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

        if (msg != NULL)
        {
            GError *err;
            gchar *debug_info;
            done = TRUE;
            waiting_eos = FALSE;
            sigint_restore();
            switch (GST_MESSAGE_TYPE(msg))
            {
                case GST_MESSAGE_ERROR:
                {
                    gst_message_parse_error(msg, &err, &debug_info);
                    GST_ERROR("Error received from element %s: %s", GST_OBJECT_NAME(msg->src), err->message);

                    std::string dinfo = debug_info ? std::string(debug_info) : "none";
                    GST_ERROR("Debugging information : %s", dinfo.c_str());

                    g_clear_error(&err);
                    g_free(debug_info);
                    ret = GST_FLOW_ERROR;
                    break;
                }
                case GST_MESSAGE_EOS:
                {
                    GST_INFO("End-Of-Stream reached");
                    ret = GST_FLOW_OK;
                    break;
                }
                default:
                {
                    // We should not reach here because we only asked for ERRORs and EOS
                    GST_WARNING("Unexpected message received %d", GST_MESSAGE_TYPE(msg));
                    ret = GST_FLOW_ERROR;
                    break;
                }
            }
            gst_message_unref(msg);
        }
        check_sigint(pipeline);
    }
    gst_object_unref(bus);
    return ret;
}

/**
 * Appsink's propose_allocation callback - Adding an GST_VIDEO_META_API_TYPE allocation meta
 *
 * @param[in] appsink               The appsink object.
 * @param[in] appsink               The allocation query.
 * @param[in] callback_data         user data.
 * @return TRUE
 * @note The adding of allocation meta is required to work with v4l2src without it copying each buffer.
 */
static gboolean appsink_propose_allocation(GstAppSink *appsink, GstQuery *query, gpointer callback_data)
{
  gst_query_add_allocation_meta(query, GST_VIDEO_META_API_TYPE, NULL);
  return TRUE;
}

/**
 * Appsink's new_sample callback
 *
 * @param[in] appsink               The appsink object.
 * @param[in] user_data             user data.
 * @return GST_FLOW_OK
 * @note Example only - only mapping the buffer to a GstMapInfo, than unmapping.
 */
static GstFlowReturn appsink_new_sample(GstAppSink *appsink, gpointer user_data) //callback func
{
  GstSample *sample;
  GstBuffer *gst_buffer;
  GstVideoFrame frame;
  hailo_media_library_buffer buffer;
  std::vector<hailo_media_library_buffer> outputs;
  hailo15_vsm vsm;
  GstVideoInfo *info = gst_video_info_new();
  MediaLibrary *media_lib = static_cast<MediaLibrary *>(user_data);
  GstFlowReturn return_status = GST_FLOW_OK;
  
  // get the incoming sample
  sample = gst_app_sink_pull_sample(appsink);
  GstCaps * caps = gst_sample_get_caps(sample);
  gst_video_info_from_caps(info, caps);

  gst_buffer = gst_sample_get_buffer(sample);
  if (gst_buffer)
  {
    // Verify buffer contains VSM metadata
    GstHailoVsmMeta *meta = reinterpret_cast<GstHailoVsmMeta *>(gst_buffer_get_meta(gst_buffer, g_type_from_name(HAILO_VSM_META_API_NAME)));
    vsm = meta->vsm;

    // frame map
    gst_video_frame_map(&frame, info, gst_buffer, GST_MAP_READ);
    create_hailo_buffer_from_video_frame(&frame,buffer,vsm);
    // frame unmap
    gst_video_frame_unmap(&frame);
  }
  gst_video_info_free(info);

  // perform vision_preproc logic
  media_library_return preproc_status = media_lib->vision_preproc->handle_frame(buffer, outputs); //send here the hailo_media_library_buffer
  if (preproc_status != MEDIA_LIBRARY_SUCCESS)
    return_status = GST_FLOW_ERROR;

  // Add detections to the frame

  hailo_media_library_buffer* new_buffer = new hailo_media_library_buffer(std::move(outputs[0]));
  media_lib->yolov8->add_frame(*new_buffer);


  gst_sample_unref(sample);
  return return_status;
}

/**
 * Create the gstreamer pipeline as string
 *
 * @return A string containing the gstreamer pipeline.
 * @note prints the return value to the stdout.
 */
std::string create_pipeline_string()
{
  std::string pipeline = "";

  pipeline = "v4l2src name=src_element num-buffers=200 device=/dev/video0 io-mode=mmap ! "
             "video/x-raw,format=NV12,width=3840,height=2160, framerate=25/1 ! "
             "queue leaky=no max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! "
             "appsink wait-on-eos=false name=hailo_sink";

  std::cout << "Pipeline:" << std::endl;
  std::cout << "gst-launch-1.0 " << pipeline << std::endl;

  return pipeline;
}

std::string create_output_pipeline_string()
{
  std::string pipeline = "";

  pipeline = "appsrc name=hailo_src ! h264parse ! rtph264pay config-interval=1 !  udpsink host=10.0.0.2 port=5000";
  
  std::cout << "Pipeline:" << std::endl;
  std::cout << "gst-launch-1.0 " << pipeline << std::endl;

  return pipeline;
}

static void appsrc_need_data(GstAppSrc *appsrc, guint size, gpointer user_data){
  MediaLibrary *media_library = (MediaLibrary*)user_data;
  std::pair<char*, uint32_t> pair;
  while(media_library->queue.empty() && !media_library->stopSignal){
    sleep(0.01);
  }
  if(media_library->stopSignal){
    if(!media_library->queue.empty()){
      delete media_library->queue.front().first;
    }
    return;
  }
  media_library->mut.lock();
  pair = media_library->queue.front();
  media_library->queue.pop();
  media_library->mut.unlock();
  char * data = (char *)pair.first;
  uint32_t size_of_data = pair.second;
  GstBuffer *gst_buffer = gst_buffer_new_allocate(nullptr, size_of_data, nullptr);
  gst_buffer_fill(gst_buffer, 0, data, size_of_data);
  GstFlowReturn ret;
  
  g_signal_emit_by_name(appsrc, "push-buffer", gst_buffer, &ret);
  if (ret != GST_FLOW_OK)
    {
        /* Something went wrong, stop pushing */
        std::cout << "ERR. ending stream" << std::endl;
        g_signal_emit_by_name(appsrc, "end-of-stream", &ret);
    }
  delete data;

}

/**
 * Set the Appsink callbacks
 *
 * @param[in] pipeline        The pipeline as a GstElement.
 * @note Sets the new_sample and propose_allocation callbacks, without callback user data (NULL).
 */
void set_callbacks_in(GstElement *pipeline, MediaLibrary * media_lib)
{
  GstAppSinkCallbacks callbacks = {NULL};

  GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline), "hailo_sink");
  callbacks.new_sample = appsink_new_sample;
  callbacks.propose_allocation = appsink_propose_allocation;

  gst_app_sink_set_callbacks(GST_APP_SINK(appsink), &callbacks, (gpointer)media_lib, NULL);
  gst_object_unref(appsink);
}

/**
 * Set the Appsrc callbacks
 *
 * @param[in] pipeline        The pipeline as a GstElement.
 */
void set_callbacks_out(GstElement *pipeline, MediaLibrary * media_lib)
{
  GstElement *appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "hailo_src");
  GstAppSrc *gstAppsrc = GST_APP_SRC(appsrc);
  // Configure the appsrc
  GstAppSrcCallbacks appsrc_callbacks = {0};
  appsrc_callbacks.need_data = appsrc_need_data;
  gst_app_src_set_callbacks(gstAppsrc, &appsrc_callbacks, (gpointer)media_lib, NULL);
  gst_object_unref(appsrc);
}

void write_encoded_data(HailoMediaLibraryBufferPtr buffer, uint32_t size, MediaLibrary* media_library)
{
  char * data = (char *)buffer->get_plane(0);
  char* new_data = new char[size];
  memcpy(new_data, data, size);
  media_library->mut.lock();
  media_library->queue.push(std::make_pair(new_data, size));
  media_library->mut.unlock();
  buffer->decrease_ref_count();
  
}

std::string read_string_from_file(const char *file_path)
{
    std::ifstream file_to_read;
    file_to_read.open(file_path);
    if (!file_to_read.is_open())
      throw std::runtime_error("config path is not valid");
    std::string file_string((std::istreambuf_iterator<char>(file_to_read)), std::istreambuf_iterator<char>());
    file_to_read.close();
    std::cout << "Read config from file: " << file_path << std::endl;
    return file_string;
}

void delete_output_file()
{
  std::ofstream fp("vision_preproc_example.h264", std::ios::out | std::ios::binary);
  if (!fp.good())
  {
    std::cout << "Error occurred at writing time!" << std::endl;
    return;
  }
  fp.close();
}



int main(int argc, char *argv[])
{
  std::string src_pipeline_string;
  GstFlowReturn ret;

  MediaLibrary * media_lib = new MediaLibrary();

  add_sigint_handler();
  delete_output_file();

  // Create and configure vision_pre_proc
  std::string preproc_config_string = read_string_from_file(VISION_PREPROC_CONFIG_FILE);
  tl::expected<MediaLibraryVisionPreProcPtr, media_library_return> vision_preproc_expected = MediaLibraryVisionPreProc::create(preproc_config_string);
  if (!vision_preproc_expected.has_value())
  {
    std::cout << "Failed to create vision_preproc" << std::endl;
    return 1;
  }

  // Init and run all media library members
  media_lib->vision_preproc = vision_preproc_expected.value();
  media_lib->yolov8_init();
  media_lib->yolov8->run();


  // Create and configure encoder
  std::string encoderosd_config_string = read_string_from_file(ENCODER_OSD_CONFIG_FILE);
  tl::expected<MediaLibraryEncoderPtr, media_library_return> encoder_expected = MediaLibraryEncoder::create(std::move(encoderosd_config_string));
  if (!encoder_expected.has_value())
  {
    std::cout << "Failed to create encoder osd" << std::endl;
    return 1;
  }
  media_lib->encoder_thread = std::async(&MediaLibrary::run_encoder_async, media_lib, media_lib->yolov8->get_queue());

  media_lib->encoder = encoder_expected.value();
  gst_init(&argc, &argv);
  std::string pipeline_string = create_pipeline_string();
  std::string output_pipeline_string = create_output_pipeline_string();
  std::cout << "Created pipeline strings." << std::endl;
  GstElement *pipeline = gst_parse_launch(pipeline_string.c_str(), NULL);
  GstElement *output_pipeline = gst_parse_launch(output_pipeline_string.c_str(), NULL);
  std::cout << "Parsed pipeline." << std::endl;
  set_callbacks_in(pipeline, media_lib);
  set_callbacks_out(output_pipeline, media_lib);
  std::cout << "Set probes and callbacks." << std::endl;
  media_lib->encoder->subscribe([&](HailoMediaLibraryBufferPtr buffer, size_t size) 
                     { write_encoded_data(buffer, size, media_lib); }); //change the write encoded data to my function
  media_lib->encoder->start();

  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  gst_element_set_state(output_pipeline, GST_STATE_PLAYING);

  ret = wait_for_end_of_pipeline(pipeline);

  media_lib->stop();
// Free resources
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_element_set_state(output_pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
  gst_object_unref(output_pipeline);

  delete media_lib;


  return ret;
}
