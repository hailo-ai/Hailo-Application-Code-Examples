#!/bin/bash
set -e

function init_variables() {
    print_help_if_needed $@
    script_dir=$(dirname $(realpath "$0"))
    source $script_dir/../../../../../scripts/misc/checks_before_run.sh --check-vaapi

    readonly RESOURCES_DIR="$TAPPAS_WORKSPACE/apps/h8/gstreamer/general/multistream_app/resources"
    readonly POSTPROCESS_DIR="$TAPPAS_WORKSPACE/apps/h8/gstreamer/libs/post_processes"
    readonly DETECTION_POSTPROCESS_SO="$POSTPROCESS_DIR/libyolo_hailortpp_post.so"
    readonly SEMSEG_POSTPROCESS_SO="$POSTPROCESS_DIR/libsemantic_segmentation.so"
    readonly HEF_PATH_DETECTION="$RESOURCES_DIR/hefs/yolov8s.hef"
    readonly HEF_PATH_SEMSEG="$RESOURCES_DIR/hefs/fcn8_resnet_v1_18.hef"

    video_sink="fpsdisplaysink video-sink=$video_sink_element text-overlay=false"
    
    detection_hef_path=$HEF_PATH_DETECTION
    detection_postprocess_so=$DETECTION_POSTPROCESS_SO
    detection_network_name="yolov8"

    semseg_hef_path=$HEF_PATH_SEMSEG
    semseg_postprocess_so=$SEMSEG_POSTPROCESS_SO

    num_of_src=4
    sources=""
    compositor_locations="sink_0::xpos=0 sink_0::ypos=0 \
	    sink_1::xpos=640 sink_1::ypos=0 \
	    sink_2::xpos=0 sink_2::ypos=320 \
	    sink_3::xpos=640 sink_3::ypos=320"

    print_gst_launch_only=false
    video_sink_element=$([ "$XV_SUPPORTED" = "true" ] && echo "xvimagesink" || echo "ximagesink")
}

function print_usage() {
    echo "Multistream Detection hailo - pipeline usage:"
    echo ""
    echo "Options:"
    echo "  --help                          Show this help"
    echo "  --show-fps                      Printing fps"
    echo "  --print-gst-launch              Print the ready gst-launch command without running it"
    exit 0
}

function print_help_if_needed() {
    while test $# -gt 0; do
        if [ "$1" = "--help" ] || [ "$1" == "-h" ]; then
            print_usage
        fi

        shift
    done
}

function parse_args() {
    while test $# -gt 0; do
        if [ "$1" = "--help" ] || [ "$1" == "-h" ]; then
            print_usage
            exit 0
        elif [ "$1" = "--print-gst-launch" ]; then
            print_gst_launch_only=true
        elif [ "$1" = "--show-fps" ]; then
            echo "Printing fps"
            additional_parameters="-v | grep hailo_display"
        else
            echo "Received invalid argument: $1. See expected arguments below:"
            print_usage
            exit 1
        fi
        shift
    done
}

function create_sources() {
    start_index=0
    identity=""

    for ((n = $start_index; n < $num_of_src; n++)); do
        sources+="filesrc location=$RESOURCES_DIR/videos/video$n.mp4 name=source_$n ! \
                qtdemux ! vaapidecodebin ! \
                queue name=hailo_preprocess_q_$n leaky=no max-size-buffers=5 max-size-bytes=0  \
                max-size-time=0 ! videoscale method=0 add-borders=false ! \
                video/x-raw,width=640,height=640,pixel-aspect-ratio=1/1 ! videocrop bottom=80 ! $identity \
                fun.sink_$n sid.src_$n ! queue name=comp_q_$n leaky=downstream max-size-buffers=10 \
                max-size-bytes=0 max-size-time=0 ! comp.sink_$n "
        streamrouter_input_streams+=" src_$n::input-streams=\"<sink_$n>\""
    done

}


function main() {
    init_variables $@
    parse_args $@
    create_sources

   NETWORK_ONE_PIPELINE="videoconvert ! videoscale qos=false method=0 add-borders=false ! \
	   video/x-raw,format=RGB,width=640,height=640,pixel-aspect-ratio=1/1 ! \
	   queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! \
	   hailonet device-count=1 vdevice-group-id=0 hef-path=$detection_hef_path is-active=true ! \
	   queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! \
	   hailofilter so-path=$detection_postprocess_so function-name=yolov8s qos=false ! \
	   queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0"

   NETWORK_TWO_PIPELINE="videoconvert ! videoscale qos=false method=0 add-borders=false ! \
	   video/x-raw,format=RGB,width=640,height=320,pixel-aspect-ratio=1/1 ! \
	   queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! \
	   hailonet device-count=1 vdevice-group-id=0 hef-path=$semseg_hef_path is-active=true ! \
	   queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! \
	   hailofilter so-path=$semseg_postprocess_so qos=false ! \
	   queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0"

    pipeline="gst-launch-1.0 \
           hailoroundrobin mode=1 name=fun ! \
	   tee name=t \
	   hailomuxer name=hmux1 \
           hailomuxer name=hmux2 \
           t. ! queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! hmux1. \
           t. ! $NETWORK_ONE_PIPELINE ! hmux1. \
           t. ! $NETWORK_TWO_PIPELINE ! hmux2. \
           hmux1. ! hmux2. \
           hmux2. ! queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! \
           hailooverlay ! \
	   hailostreamrouter name=sid $streamrouter_input_streams compositor name=comp start-time-selection=0 $compositor_locations ! \
	   queue name=hailo_video_q_0 leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \
	   videoconvert ! queue name=hailo_display_q_0 leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \
	   fpsdisplaysink video-sink=$video_sink_element name=hailo_display sync=false text-overlay=false \
	   $sources ${additional_parameters}"

    echo ${pipeline}
    if [ "$print_gst_launch_only" = true ]; then
        exit 0
    fi

    echo "Running Pipeline..."
    eval "${pipeline}"

}

main $@
