#!/bin/bash
set -e

function init_variables() {
    print_help_if_needed $@

    script_dir=$(dirname $(realpath "$0"))
    source $script_dir/../../../../../scripts/misc/checks_before_run.sh

    readonly POSTPROCESS_DIR="$TAPPAS_WORKSPACE/apps/h8/gstreamer/libs/post_processes/"
    readonly RESOURCES_DIR="$TAPPAS_WORKSPACE/apps/h8/gstreamer/general/detection/resources"

    readonly DEFAULT_POSTPROCESS_SO="$POSTPROCESS_DIR/libyolo_hailortpp_post.so"
    readonly DEFAULT_NETWORK_NAME="yolov8m"
    readonly DEFAULT_BATCH_SIZE="1"
    readonly DEFAULT_VIDEO_SOURCE="$RESOURCES_DIR/detection.mp4"
    readonly DEFAULT_HEF_PATH="$RESOURCES_DIR/yolov8m.hef"


    video_sink_element=$([ "$XV_SUPPORTED" = "true" ] && echo "xvimagesink" || echo "ximagesink")
    video_sink="fpsdisplaysink video-sink=$video_sink_element text-overlay=false"
    postprocess_so=$DEFAULT_POSTPROCESS_SO
    network_name=$DEFAULT_NETWORK_NAME
    input_source=$DEFAULT_VIDEO_SOURCE
    batch_size=$DEFAULT_BATCH_SIZE
    hef_path=$DEFAULT_HEF_PATH
    json_config_path="null"
    nms_score_threshold=0.3 
    nms_iou_threshold=0.45

    thresholds_str="nms-score-threshold=${nms_score_threshold} nms-iou-threshold=${nms_iou_threshold} output-format-type=HAILO_FORMAT_TYPE_FLOAT32"

    print_gst_launch_only=false
    additional_parameters=""
    stats_element=""
    debug_stats_export=""
    sync_pipeline=false
    device_id_prop=""
    tappas_gui_mode=false
}

function print_help_if_needed() {
    while test $# -gt 0; do
        if [ "$1" = "--help" ] || [ "$1" == "-h" ]; then
            print_usage
        fi

        shift
    done
}

function print_usage() {
    echo "Detection pipeline usage:"
    echo ""
    echo "Options:"
    echo "  -h --help                  Show this help"
    echo "  --network NETWORK          Set network to use. choose from [yolov5, mobilenet_ssd, nanodet, yolov8], default is yolov8"
    echo "  -i INPUT --input INPUT     Set the input source (default $input_source)"
    echo "  --show-fps                 Print fps"
    echo "  --print-gst-launch         Print the ready gst-launch command without running it"
    echo "  --print-device-stats       Print the power and temperature measured"
    echo "  --tcp-address              Used for TAPPAS GUI, switchs the sink to TCP client"
    exit 0
}

function parse_args() {
    while test $# -gt 0; do
        if [ $1 == "--network" ]; then
            if [ $2 == "mobilenet_ssd" ]; then
                network_name="mobilenet_ssd"
                batch_size="4"
                hef_path="$RESOURCES_DIR/ssd_mobilenet_v1.hef"
                postprocess_so="$POSTPROCESS_DIR/libmobilenet_ssd_post.so"
                json_config_path="null"
                thresholds_str="output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
            elif [ $2 == "nanodet" ]; then
                network_name="nanodet_repvgg"
                hef_path="$RESOURCES_DIR/nanodet_repvgg.hef"
                postprocess_so="$POSTPROCESS_DIR/libnanodet_post.so"
                json_config_path="null"
                thresholds_str=""
            elif [ $2 == "yolov5" ]; then
                network_name="yolov5"
                hef_path="$RESOURCES_DIR/yolov5m_wo_spp_60p.hef"
            elif [ $2 != "yolov8" ]; then
                echo "Received invalid network: $2. See expected arguments below:"
                print_usage
                exit 1
            fi
            shift
        elif [ "$1" = "--print-gst-launch" ]; then
            print_gst_launch_only=true
        elif [ "$1" = "--print-device-stats" ]; then
            hailo_bus_id=$(hailortcli scan | awk '{ print $NF }' | tail -n 1)
            device_id_prop="device_id=$hailo_bus_id"
            stats_element="hailodevicestats $device_id_prop"
            debug_stats_export="GST_DEBUG=hailodevicestats:5"
        elif [ "$1" = "--show-fps" ]; then
            echo "Printing fps"
            additional_parameters="-v | grep -e hailo_display -e hailodevicestats"
        elif [ "$1" = "--input" ] || [ "$1" == "-i" ]; then
            input_source="$2"
            shift
        elif [ "$1" = "--tcp-address" ]; then
            tcp_host=$(echo $2 | awk -F':' '{print $1}')
            tcp_port=$(echo $2 | awk -F':' '{print $2}')
            video_sink="queue name=queue_before_sink leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \
                        videoscale ! video/x-raw,width=836,height=546,format=RGB ! \
                        tcpclientsink host=$tcp_host port=$tcp_port"
            shift
        else
            echo "Received invalid argument: $1. See expected arguments below:"
            print_usage
            exit 1
        fi

        shift
    done
}

init_variables $@
parse_args $@

# If the video provided is from a camera
if [[ $input_source =~ "/dev/video" ]]; then
    source_element="v4l2src device=$input_source name=src_0 ! videoflip video-direction=horiz"
else
    source_element="filesrc location=$input_source name=src_0 ! decodebin"
fi

PIPELINE="${debug_stats_export} gst-launch-1.0 ${stats_element} \
    $source_element ! \
    queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \
    videoscale qos=false n-threads=2 ! video/x-raw, pixel-aspect-ratio=1/1 ! \
    queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \
    videoconvert n-threads=2 qos=false ! \
    queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \
    hailonet hef-path=$hef_path $device_id_prop batch-size=$batch_size $thresholds_str ! \
    queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \
    hailofilter function-name=$network_name so-path=$postprocess_so config-path=$json_config_path qos=false ! \
    queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \
    hailopython qos=false module=tracker.py ! \
    queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \
    hailooverlay qos=false ! \
    queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \
    videoconvert n-threads=2 qos=false ! \
    queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \
    x264enc ! mp4mux ! filesink location=output.mp4"   
    
#    $video_sink name=hailo_display sync=$sync_pipeline ${additional_parameters}"

echo "Running $network_name"
echo ${PIPELINE}

if [ "$print_gst_launch_only" = true ]; then
    exit 0
fi

eval ${PIPELINE}

