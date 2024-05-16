#!/bin/bash
set -e

CURRENT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

function init_variables() {
    readonly DEFAULT_ENCODER="h264"
    readonly DEFAULT_PORT="5000"
    readonly DEFAULT_ADDRESS="10.0.0.2"

    port=$DEFAULT_PORT
    address=$DEFAULT_ADDRESS
    encoder=$DEFAULT_ENCODER

    num_cores_to_use=$(($(nproc) / 2))
    queue_params="max-size-buffers=30 max-size-bytes=0 max-size-time=0"
    queue_params_non_leaky="${queue_params} leaky=no"
    queue_params_leaky="${queue_params} leaky=downstream"

    print_gst_launch_only=false
    additional_parameters=""
}

function print_usage() {
    echo "UDP stream display pipeline usage:"
    echo ""
    echo "Options:"
    echo "  --help                        Show this help"
    echo "  -p PORT --port PORT           Set the port source (default $port)"
    echo "  -a ADDRESS --address ADDRESS  Set the address source (default $address)"
    echo "  -e ENCODER --encoder ENCODER  Select the encoder type choose from [h265, h264]. default $encoder"
    echo "  --show-fps                    Print fps"
    echo "  --print-gst-launch            Print the ready gst-launch command without running it"
    exit 0
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
        elif [ "$1" = "--port" ] || [ "$1" = "-p" ]; then
            port="$2"
            shift
        elif [ "$1" = "--address" ] || [ "$1" = "-a" ]; then
            address="$2"
            shift
        elif [ "$1" = "--encoder" ] || [ "$1" = "-e" ]; then
            encoder="$2"
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


PIPELINE="gst-launch-1.0 \
    udpsrc port=$port address=$address ! application/x-rtp,encoding-name=${encoder^^} ! \
    queue ${queue_params_non_leaky} ! rtpjitterbuffer mode=0 !  \
    queue ${queue_params_non_leaky} ! rtp${encoder}depay !  \
    queue ${queue_params_non_leaky} ! ${encoder}parse ! avdec_${encoder} ! \
    queue ${queue_params_leaky} ! \
    videoconvert n-threads=${num_cores_to_use} ! queue ${queue_params_non_leaky} ! \
    fpsdisplaysink name=hailo_display text-overlay=false sync=false ${additional_parameters}"

echo ${PIPELINE}

if [ "$print_gst_launch_only" = true ]; then
    exit 0
fi

eval ${PIPELINE}
