import os
import re
batch_size = 8
video_sink = "xvimagesink"

# Note: only 16:9 resolutions are supported
RES_X = {1280}
RES_Y = {720}

def get_pipeline_multi(current_path, detector_pipeline, sync, input_uri, tappas_workspace, tapppas_version):
    # Initialize directories and paths
    RESOURCES_DIR = os.path.join(current_path, "resources")
    POSTPROCESS_DIR = os.path.join(tappas_workspace, "apps/h8/gstreamer/libs/post_processes")
    STREAM_ID_SO = os.path.join(POSTPROCESS_DIR, "libstream_id_tool.so")
    ADD_STREAM_ID_PATH = os.path.join(current_path, "add_stream_id.py")
    hailopython_path = os.path.join(current_path, "clip_app/clip_hailopython_multi.py")
    
    if (detector_pipeline == "fast_sam"):    
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
        YOLO5_HEF_PATH = os.path.join(RESOURCES_DIR, "yolov5s_personface.hef")
        YOLO5_CONFIG_PATH = os.path.join(RESOURCES_DIR, "configs/yolov5_personface.json")
        DETECTION_POST_PIPE = f'hailofilter so-path={YOLO5_POSTPROCESS_SO} qos=false function_name={YOLO5_NETWORK_NAME} config-path={YOLO5_CONFIG_PATH} '
        hef_path = YOLO5_HEF_PATH

    # CLIP 
    clip_hef_path = os.path.join(RESOURCES_DIR, "clip_resnet_50x4.hef")
    clip_postprocess_so = os.path.join(RESOURCES_DIR, "libclip_post.so")
    DEFAULT_CROP_SO = os.path.join(RESOURCES_DIR, "libclip_croppers.so")

    DEFAULT_VDEVICE_KEY = "1"
    
    # define multi sources pipeline
    video_sources = [os.path.join(RESOURCES_DIR,'reid0.mp4'),
                     os.path.join(RESOURCES_DIR,'reid1.mp4'),
                     os.path.join(RESOURCES_DIR,'reid2.mp4'),
                     os.path.join(RESOURCES_DIR,'reid3.mp4')]
    
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
    
    DETECTION_PIPELINE = f'{QUEUE()} name=pre_detecion_net ! \
        video/x-raw, pixel-aspect-ratio=1/1 ! \
        hailonet hef-path={hef_path} batch-size={batch_size} vdevice-key={DEFAULT_VDEVICE_KEY} \
        multi-process-service=true scheduler-timeout-ms=100 scheduler-priority=31 ! \
        {QUEUE()} name=pre_detecion_post ! \
        {DETECTION_POST_PIPE} ! \
        {QUEUE()}'
    
    CLIP_PIPELINE = f'{QUEUE()} name=pre_clip_net ! \
        hailonet hef-path={clip_hef_path} batch-size={batch_size} vdevice-key={DEFAULT_VDEVICE_KEY} \
        multi-process-service=true scheduler-timeout-ms=1000 ! \
        {QUEUE()} ! \
        hailofilter so-path={clip_postprocess_so} qos=false ! \
        {QUEUE()}'

    if detector_pipeline == "person":
        class_id = 1
        crop_function_name = "person_cropper"
    elif detector_pipeline == "face":
        class_id = 2
        crop_function_name = "face_cropper"
    else: # fast_sam
        class_id = 0
        crop_function_name = "object_cropper"
    TRACKER = f'hailotracker name=hailo_tracker class-id={class_id} kalman-dist-thr=0.8 iou-thr=0.8 init-iou-thr=0.7 \
                keep-new-frames=2 keep-tracked-frames=35 keep-lost-frames=2 keep-past-metadata=true qos=false ! \
                {QUEUE()} '
    
    # DETECTION_PIPELINE_MUXER = f'{QUEUE(buffer_size=12, name="pre_detection_tee")} max-size-buffers=12 ! tee name=detection_t hailomuxer name=hmux \
    #     detection_t. ! {QUEUE(buffer_size=20, name="detection_bypass_q")} ! hmux.sink_0 \
    #     detection_t. ! {DETECTION_PIPELINE} ! hmux.sink_1 \
    #     hmux. ! {QUEUE()} '
    
    WHOLE_BUFFER_CROP_SO = os.path.join(POSTPROCESS_DIR, "cropping_algorithms/libwhole_buffer.so")
    
    DETECTION_PIPELINE_MUXER = f'{QUEUE(buffer_size=12, name="pre_detection_tee")} max-size-buffers=12 ! \
        hailocropper  name=detection_crop so-path={WHOLE_BUFFER_CROP_SO} function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true \
        hailoaggregator name=agg1 \
        detection_crop. ! {QUEUE(buffer_size=20, name="detection_bypass_q")} ! agg1.sink_0 \
        detection_crop. ! {DETECTION_PIPELINE} ! agg1.sink_1 \
        agg1. ! {QUEUE()} '

    if detector_pipeline == "none":
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
    CLIP_DISPLAY_PIPELINE = f'{QUEUE()} ! \
                            fpsdisplaysink name=hailo_display video-sink={video_sink} sync={sync} text-overlay=true '
    
    def CLIP_SMALL_DISPLAY_PIPELINE(i):
        return f'{QUEUE()} ! videoscale ! \
                 {QUEUE()} ! video/x-raw, width=480, height=270 ! \
                 fpsdisplaysink name=hailo_display_{i} sync={sync} text-overlay=true '
    
    STREAM_ROUTER_PIPELINE = f'hailostreamrouter name=sid '
    STREAM_ROUTER_PIPELINE_END = ''
    STREAM_ROUTER_PIPELINE_END += f'input-selector name=input_selector '
    for i in range(len(video_sources)):
        STREAM_ROUTER_PIPELINE += f'src_{i}::input-streams="<sink_{i}>" '
        STREAM_ROUTER_PIPELINE_END += f'sid.src_{i} ! tee name=display_t_{i} ! {CLIP_SMALL_DISPLAY_PIPELINE(i)} '
        STREAM_ROUTER_PIPELINE_END += f'display_t_{i}. ! queue leaky=downstream max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! input_selector.sink_{i} '
    STREAM_ROUTER_PIPELINE_END += f'input_selector. '
    STREAM_ROUTER_PIPELINE += STREAM_ROUTER_PIPELINE_END

    # Text to image matcher
    CLIP_POSTPROCESS_PIPELINE = f' hailopython name=pyproc module={hailopython_path} qos=false ! \
        {QUEUE()} ! \
        hailooverlay local-gallery=false show-confidence=true font-thickness=2 qos=false ! \
        {QUEUE()} ! \
        videoconvert name=videoconvert_overlay n-threads=2 qos=false ! video/x-raw, format=YV12 '

    SOURCE_PIPELINE = 'hailoroundrobin name=rr_arbiter mode=2 '

    for i in range(len(video_sources)):
        SOURCE_PIPELINE += f'uridecodebin uri=file://{video_sources[i]} name=uridecodebon_{i} ! '
        SOURCE_PIPELINE += f'{QUEUE()} name=src_q{i} ! videoconvert name=convert_src_{i} qos=false ! '
        SOURCE_PIPELINE += f'hailofilter name=set_id_{i} so-path={STREAM_ID_SO} config-path=SRC_{i} qos=false ! '
        SOURCE_PIPELINE += f'{QUEUE()} name=scale_q{i} ! videoscale name=scale_src_{i} qos=false ! '
        SOURCE_PIPELINE += f'video/x-raw, width={RES_X}, height={RES_Y}, format=RGB ! '
        SOURCE_PIPELINE += f'rr_arbiter.sink_{i} '
    SOURCE_PIPELINE += ' rr_arbiter. '
    
    # PIPELINE
    if detector_pipeline == "none":
        PIPELINE = f'{SOURCE_PIPELINE} ! \
            {CLIP_MUXER_PIPELINE} ! \
            {CLIP_POSTPROCESS_PIPELINE} ! \
            {STREAM_ROUTER_PIPELINE}'
    else:
        PIPELINE = f'{SOURCE_PIPELINE} ! \
            {DETECTION_PIPELINE_WRAPPER} ! \
            {TRACKER} ! \
            {CLIP_CROPPER_PIPELINE} ! \
            {CLIP_POSTPROCESS_PIPELINE} ! \
            {STREAM_ROUTER_PIPELINE} ! \
            {CLIP_DISPLAY_PIPELINE}'

    return PIPELINE
