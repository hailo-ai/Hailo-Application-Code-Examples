import os
import re
batch_size = 8
video_sink = "xvimagesink"
# Note: only 16:9 resolutions are supported
# RES_X = 1920
# RES_Y = 1080
RES_X = {1280}
RES_Y = {720}

# Check Hailo Device Type from the environment variable DEVICE_ARCHITECTURE
# If the environment variable is not set, default to HAILO8L
device_architecture = os.getenv("DEVICE_ARCHITECTURE")
if device_architecture is None or device_architecture == "HAILO8L":
    device_architecture = "HAILO8L"
    # HEF files for H8L
    YOLO5_HEF_NAME = "yolov5s_personface_h8l_pi.hef"
    CLIP_HEF_NAME = "clip_resnet_50x4_h8l.hef"
else:
    device_architecture = "HAILO8"
    # HEF files for H8
    YOLO5_HEF_NAME = "yolov5s_personface.hef"
    CLIP_HEF_NAME = "clip_resnet_50x4.hef"

def get_pipeline(self):
    # Initialize directories and paths
    RESOURCES_DIR = os.path.join(self.current_path, "resources")
    POSTPROCESS_DIR = self.tappas_postprocess_dir
    hailopython_path = os.path.join(self.current_path, "clip_app/clip_hailopython.py")
    if self.detector == "fast_sam":
        # FASTSAM
        # DETECTION_HEF_PATH = os.path.join(RESOURCES_DIR, "fast_sam_s.hef")
        DETECTION_HEF_PATH = os.path.join(RESOURCES_DIR, "yolov8s_fastsam_single_context.hef")
        DETECTION_POST = os.path.join(RESOURCES_DIR, "libfastsam_post.so")
        DETECTION_POST_PIPE = f'hailofilter so-path={DETECTION_POST} qos=false '
        hef_path = DETECTION_HEF_PATH
    else:
        # personface
        YOLO5_POSTPROCESS_SO = os.path.join(POSTPROCESS_DIR, "libyolo_post.so")
        YOLO5_NETWORK_NAME = "yolov5_personface_letterbox"
        YOLO5_HEF_PATH = os.path.join(RESOURCES_DIR, YOLO5_HEF_NAME)
        YOLO5_CONFIG_PATH = os.path.join(RESOURCES_DIR, "configs/yolov5_personface.json")
        DETECTION_POST_PIPE = f'hailofilter so-path={YOLO5_POSTPROCESS_SO} qos=false function_name={YOLO5_NETWORK_NAME} config-path={YOLO5_CONFIG_PATH} '
        hef_path = YOLO5_HEF_PATH

    # CLIP
    clip_hef_path = os.path.join(RESOURCES_DIR, CLIP_HEF_NAME)
    clip_postprocess_so = os.path.join(RESOURCES_DIR, "libclip_post.so")
    DEFAULT_CROP_SO = os.path.join(RESOURCES_DIR, "libclip_croppers.so")
    clip_matcher_so = os.path.join(RESOURCES_DIR, "libclip_matcher.so")
    clip_matcher_config = os.path.join(self.current_path, "embeddings.json")
    DEFAULT_VDEVICE_KEY = "1"

    def QUEUE(name=None, buffer_size=3, name_suffix=""):
        q_str = f'queue leaky=no max-size-buffers={buffer_size} max-size-bytes=0 max-size-time=0 silent=true '
        if name is not None:
            q_str += f'name={name}{name_suffix} '
        return q_str

    # Debug display
    DISPLAY_PROBE = f'tee name=probe_tee ! \
        {QUEUE()} ! videoconvert ! autovideosink name=probe_display sync=false \
        probe_tee. ! {QUEUE()}'

    RATE_PIPELINE = f' {QUEUE()} name=rate_queue ! video/x-raw, framerate=30/1 '
    # Check if the input seems like a v4l2 device path (e.g., /dev/video0)
    sync = False # sync_req is relevant only when working with video files
    if re.match(r'/dev/video\d+', self.input_uri):
        SOURCE_PIPELINE = f'v4l2src device={self.input_uri} ! image/jpeg, framerate=30/1 ! decodebin ! {QUEUE()} ! \
        videoconvert ! {QUEUE()} ! videoscale ! video/x-raw, width={RES_X}, height={RES_Y}, format=RGB ! {QUEUE()} ! videoflip video-direction=horiz ! '
    elif re.match(r'0x\w+', self.input_uri): # Window ID - get from xwininfo
        SOURCE_PIPELINE = f"ximagesrc xid={self.input_uri} ! {QUEUE()} ! videoscale ! "
    else:
        sync = self.sync_req
        # convert the file to a uri
        uri = os.path.abspath(self.input_uri)
        uri = f'file://{uri}'
        SOURCE_PIPELINE = f"uridecodebin uri={uri} ! {QUEUE()} ! videoscale ! "
    SOURCE_PIPELINE += f'{QUEUE()} name=src_convert_queue ! videoconvert n-threads=2 ! video/x-raw, width={RES_X}, height={RES_Y}, format=RGB '

    DETECTION_PIPELINE = f'{QUEUE()} name=pre_detection_scale ! videoscale n-threads=4 qos=false ! \
        {QUEUE()} name=pre_detection_net ! \
        video/x-raw, pixel-aspect-ratio=1/1 ! \
        hailonet hef-path={hef_path} batch-size={batch_size} vdevice-key={DEFAULT_VDEVICE_KEY} \
        multi-process-service=false scheduler-timeout-ms=100 scheduler-priority=31 ! \
        {QUEUE()} name=pre_detection_post ! \
        {DETECTION_POST_PIPE} ! \
        {QUEUE()}'


    CLIP_PIPELINE = f'{QUEUE()} name=pre_clip_net ! \
        hailonet hef-path={clip_hef_path} batch-size={batch_size} vdevice-key={DEFAULT_VDEVICE_KEY} \
        multi-process-service=false scheduler-timeout-ms=1000 ! \
        {QUEUE()} ! \
        hailofilter so-path={clip_postprocess_so} qos=false ! \
        {QUEUE()}'

    if self.detector == "person":
        class_id = 1
        crop_function_name = "person_cropper"
    elif self.detector == "face":
        class_id = 2
        crop_function_name = "face_cropper"
    else: # fast_sam
        class_id = 0
        crop_function_name = "object_cropper"
    TRACKER = f'hailotracker name=hailo_tracker class-id={class_id} kalman-dist-thr=0.8 iou-thr=0.9 init-iou-thr=0.7 \
                keep-new-frames=2 keep-tracked-frames=15 keep-lost-frames=2 keep-past-metadata=true qos=false ! \
                {QUEUE()} '

    WHOLE_BUFFER_CROP_SO = os.path.join(POSTPROCESS_DIR, "cropping_algorithms/libwhole_buffer.so")

    DETECTION_PIPELINE_MUXER = f'{QUEUE(buffer_size=12, name="pre_detection_tee")} max-size-buffers=12 ! \
        hailocropper  name=detection_crop so-path={WHOLE_BUFFER_CROP_SO} function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true \
        hailoaggregator name=agg1 \
        detection_crop. ! {QUEUE(buffer_size=20, name="detection_bypass_q")} ! agg1.sink_0 \
        detection_crop. ! {DETECTION_PIPELINE} ! agg1.sink_1 \
        agg1. ! {QUEUE()} '

    if self.detector == "none":
        DETECTION_PIPELINE_WRAPPER = ""
    else:
        DETECTION_PIPELINE_WRAPPER = DETECTION_PIPELINE_MUXER

    # Clip pipeline with cropper integration
    CLIP_CROPPER_PIPELINE = f'hailocropper so-path={DEFAULT_CROP_SO} function-name={crop_function_name} \
        use-letterbox=true no-scaling-bbox=true internal-offset=true name=cropper \
        hailoaggregator name=agg \
        cropper. ! {QUEUE(buffer_size=20, name="clip_bypass_q")} ! agg.sink_0 \
        cropper. ! {CLIP_PIPELINE} ! agg.sink_1 \
        agg. ! {QUEUE()} '

    # Clip pipeline with muxer integration (no cropper)
    CLIP_MUXER_PIPELINE = f'tee name=clip_t hailomuxer name=clip_hmux \
        clip_t. ! {QUEUE(buffer_size=20, name="clip_bypass_q")} ! clip_hmux.sink_0 \
        clip_t. ! {QUEUE()} ! videoscale n-threads=4 qos=false ! {CLIP_PIPELINE} ! clip_hmux.sink_1 \
        clip_hmux. ! {QUEUE()} '

    # Display pipelines
    CLIP_DISPLAY_PIPELINE = f'{QUEUE()} ! videoconvert n-threads=2 ! \
                            fpsdisplaysink name=hailo_display video-sink={video_sink} sync={sync} text-overlay={self.show_fps} '

    # Text to image matcher
    CLIP_PYTHON_MATCHER = f'hailopython name=pyproc module={hailopython_path} qos=false '
    CLIP_CPP_MATCHER = f'hailofilter so-path={clip_matcher_so} qos=false config-path={clip_matcher_config} '

    CLIP_POSTPROCESS_PIPELINE = f' {CLIP_PYTHON_MATCHER} ! \
        {QUEUE()} ! \
        identity name=identity_callback ! \
        {QUEUE()} ! \
        hailooverlay local-gallery=false show-confidence=true font-thickness=2 qos=false '

    # PIPELINE
    if self.detector == "none":
        PIPELINE = f'{SOURCE_PIPELINE} ! \
        {CLIP_MUXER_PIPELINE} ! \
        {CLIP_POSTPROCESS_PIPELINE} ! \
	    {CLIP_DISPLAY_PIPELINE}'
    else:
        PIPELINE = f'{SOURCE_PIPELINE} ! \
        {DETECTION_PIPELINE_WRAPPER} ! \
        {TRACKER} ! \
        {CLIP_CROPPER_PIPELINE} ! \
        {CLIP_POSTPROCESS_PIPELINE} ! \
		{CLIP_DISPLAY_PIPELINE}'

    return PIPELINE
