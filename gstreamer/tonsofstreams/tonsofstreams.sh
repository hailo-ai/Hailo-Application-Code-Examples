#!/bin/bash
set -e

function init_variables() {
    print_help_if_needed $@
    script_dir=$(dirname $(realpath "$0"))
    source $TAPPAS_WORKSPACE/scripts/misc/checks_before_run.sh
    
    readonly TAPPAS_VERSION=$(grep -a1 project $TAPPAS_WORKSPACE/core/hailo/meson.build | grep version | cut -d':' -f2 | tr -d "', ")
    # check if version is 3.24.0
    if [[ "$TAPPAS_VERSION" != "3.24.0" ]]; then
        echo "Error: TAPPAS version is not 3.24.0. Please use TAPPAS 3.24.0" >&2
        exit 1
    fi
    
    readonly RESOURCES_DIR="$script_dir/resources"
    # note that in older TAPPAS version the h8 lib should be ommited
    readonly POSTPROCESS_DIR="$TAPPAS_WORKSPACE/apps/h8/gstreamer/libs/post_processes/"
    readonly POSTPROCESS_SO="$POSTPROCESS_DIR/libyolo_post.so"
    readonly HEF_PATH="$RESOURCES_DIR/hefs/yolov5m_wo_spp_60p.hef"
    readonly DEFAULT_JSON_CONFIG_PATH="$RESOURCES_DIR/configs/yolov5.json" 
    readonly BYTE_STREAMS_DIR="$RESOURCES_DIR/video/bytestreams/"
    
    readonly QUEUE="queue leaky=no max-size-bytes=0 max-size-time=0"
    num_of_src=50
    num_of_devices=4
    additonal_parameters=""
    sources=""
    compositor_locations_50="sink_0::xpos=0 sink_0::ypos=0 sink_1::xpos=320 sink_1::ypos=0 sink_2::xpos=640 sink_2::ypos=0 sink_3::xpos=960 sink_3::ypos=0 sink_4::xpos=1280 sink_4::ypos=0 sink_5::xpos=1600 sink_5::ypos=0 sink_6::xpos=1920 sink_6::ypos=0 sink_7::xpos=2240 sink_7::ypos=0 sink_8::xpos=2560 sink_8::ypos=0 sink_9::xpos=2880 sink_9::ypos=0 sink_10::xpos=0 sink_10::ypos=320 sink_11::xpos=320 sink_11::ypos=320 sink_12::xpos=640 sink_12::ypos=320 sink_13::xpos=960 sink_13::ypos=320 sink_14::xpos=1280 sink_14::ypos=320 sink_15::xpos=1600 sink_15::ypos=320 sink_16::xpos=1920 sink_16::ypos=320 sink_17::xpos=2240 sink_17::ypos=320 sink_18::xpos=2560 sink_18::ypos=320 sink_19::xpos=2880 sink_19::ypos=320 sink_20::xpos=0 sink_20::ypos=640 sink_21::xpos=320 sink_21::ypos=640 sink_22::xpos=640 sink_22::ypos=640 sink_23::xpos=960 sink_23::ypos=640 sink_24::xpos=1280 sink_24::ypos=640 sink_25::xpos=1600 sink_25::ypos=640 sink_26::xpos=1920 sink_26::ypos=640 sink_27::xpos=2240 sink_27::ypos=640 sink_28::xpos=2560 sink_28::ypos=640 sink_29::xpos=2880 sink_29::ypos=640 sink_30::xpos=0 sink_30::ypos=960 sink_31::xpos=320 sink_31::ypos=960 sink_32::xpos=640 sink_32::ypos=960 sink_33::xpos=960 sink_33::ypos=960 sink_34::xpos=1280 sink_34::ypos=960 sink_35::xpos=1600 sink_35::ypos=960 sink_36::xpos=1920 sink_36::ypos=960 sink_37::xpos=2240 sink_37::ypos=960 sink_38::xpos=2560 sink_38::ypos=960 sink_39::xpos=2880 sink_39::ypos=960 sink_40::xpos=0 sink_40::ypos=1280 sink_41::xpos=320 sink_41::ypos=1280 sink_42::xpos=640 sink_42::ypos=1280 sink_43::xpos=960 sink_43::ypos=1280 sink_44::xpos=1280 sink_44::ypos=1280 sink_45::xpos=1600 sink_45::ypos=1280 sink_46::xpos=1920 sink_46::ypos=1280 sink_47::xpos=2240 sink_47::ypos=1280 sink_48::xpos=2560 sink_48::ypos=1280 sink_49::xpos=2880 sink_49::ypos=1280 "
    compositor_locations_32='sink_0::xpos=0 sink_0::ypos=0 sink_1::xpos=320 sink_1::ypos=0 sink_2::xpos=640 sink_2::ypos=0 sink_3::xpos=960 sink_3::ypos=0 sink_4::xpos=1280 sink_4::ypos=0 sink_5::xpos=1600 sink_5::ypos=0 sink_6::xpos=1920 sink_6::ypos=0 sink_7::xpos=2240 sink_7::ypos=0 sink_8::xpos=0 sink_8::ypos=320 sink_9::xpos=320 sink_9::ypos=320 sink_10::xpos=640 sink_10::ypos=320 sink_11::xpos=960 sink_11::ypos=320 sink_12::xpos=1280 sink_12::ypos=320 sink_13::xpos=1600 sink_13::ypos=320 sink_14::xpos=1920 sink_14::ypos=320 sink_15::xpos=2240 sink_15::ypos=320 sink_16::xpos=0 sink_16::ypos=640 sink_17::xpos=320 sink_17::ypos=640 sink_18::xpos=640 sink_18::ypos=640 sink_19::xpos=960 sink_19::ypos=640 sink_20::xpos=1280 sink_20::ypos=640 sink_21::xpos=1600 sink_21::ypos=640 sink_22::xpos=1920 sink_22::ypos=640 sink_23::xpos=2240 sink_23::ypos=640 sink_24::xpos=0 sink_24::ypos=960 sink_25::xpos=320 sink_25::ypos=960 sink_26::xpos=640 sink_26::ypos=960 sink_27::xpos=960 sink_27::ypos=960 sink_28::xpos=1280 sink_28::ypos=960 sink_29::xpos=1600 sink_29::ypos=960 sink_30::xpos=1920 sink_30::ypos=960 sink_31::xpos=2240 sink_31::ypos=960 '
    compositor_locations=$compositor_locations_32
    print_gst_launch_only=false
    video_sink_element=$([ "$XV_SUPPORTED" = "true" ] && echo "xvimagesink" || echo "ximagesink")
    json_config_path=$DEFAULT_JSON_CONFIG_PATH     
}

function print_usage() {
    echo "Multistream Detection hailo - pipeline usage:"
    echo ""
    echo "Options:"
    echo "  --help                          Show this help"
    echo "  --show-fps                      Printing fps"
    echo "  --num-of-sources NUM            Setting number of sources to given input (default value is 12, maximum value is 16)"
    echo "  --print-gst-launch              Print the ready gst-launch command without running it"
    echo "  --num-of-devices NUM            Setting number Hailo devices to use (default value is 4)"
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
            additonal_parameters="-v 2>&1 | grep hailo_display"
        elif [ "$1" = "--num-of-sources" ]; then
            shift
            echo "Setting number of sources to $1"
            num_of_src=$1
            if (( $num_of_src > 32 )); then
                compositor_locations=$compositor_locations_50
            fi
       elif [ "$1" = "--num-of-devices" ]; then
            shift
            echo "Setting number of devices to $1"
            num_of_devices=$1
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
    
    for ((n = $start_index; n < $num_of_src; n++)); do
        sources+="multifilesrc location=$BYTE_STREAMS_DIR/bytestream$n.mp4 loop=true name=source_$n ! decodebin ! \
                videorate ! video/x-raw,framerate=20/1 ! \
                $QUEUE name=hailo_preprocess_q_$n max-size-buffers=5 ! \
                videoconvert ! videoscale method=0 add-borders=false qos=false ! \
                video/x-raw,width=640,height=640,pixel-aspect-ratio=1/1 ! fun.sink_$n \
                sid.src_$n ! $QUEUE name=comp_q_$n max-size-buffers=5 ! comp.sink_$n "
    done
}

function main() {
    init_variables $@
    parse_args $@
    create_sources

    pipeline="$debug_init gst-launch-1.0 \
    hailoroundrobin name=fun ! \
    $QUEUE name=hailo_pre_infer_q_0 max-size-buffers=32 ! \
    hailonet hef-path=$HEF_PATH is-active=true device-count=$num_of_devices batch-size=16 ! \
    $QUEUE name=hailo_postprocess0 max-size-buffers=32 ! \
    hailofilter so-path=$POSTPROCESS_SO config-path=$json_config_path qos=false ! \
    $QUEUE name=hailo_draw0 leaky=no max-size-buffers=32 ! \
    videoscale method=0 add-borders=false qos=false ! \
    video/x-raw,width=320,height=320,pixel-aspect-ratio=1/1 ! \
    hailooverlay ! \
    streamiddemux name=sid \
    compositor name=comp start-time-selection=0 $compositor_locations ! \
    queue name=hailo_video_q_0 leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! \
    videoconvert ! \
    $QUEUE name=hailo_display_q_0 max-size-buffers=3 ! \
    fpsdisplaysink video-sink=$video_sink_element name=hailo_display sync=false text-overlay=false \
    $sources ${additonal_parameters}"
    # fpsdisplaysink video-sink=$video_sink_element name=hailo_display sync=false text-overlay=false \
    # video_sink_element name=hailo_display sync=false 
    echo ${pipeline}
    if [ "$print_gst_launch_only" = true ]; then
        exit 0
    fi

    echo "Running Pipeline..."
    eval "${pipeline}"

}

main $@
