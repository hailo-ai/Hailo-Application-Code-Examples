#!/bin/bash

declare -A COMPILER=( [x86_64]=/usr/bin/gcc
                      [aarch64]=/usr/bin/aarch64-linux-gnu-gcc
                      [armv7l]=/usr/bin/arm-linux-gnueabi-gcc )

# HAILORT_ROOT=/local/users/omerw/HailoRT/4.4.0/Installer/platform/hailort

for ARCH in x86_64
do
    echo "-I- Building ${ARCH}"
    mkdir -p build/${ARCH}
    cmake -H. -Bbuild/${ARCH} -DARCH=${ARCH} -DCMAKE_C_COMPILER=${COMPILER[${ARCH}]} -DCMAKE_PREFIX_PATH=/local/users/omerw/.local/share/cmake
    cmake --build build/${ARCH}
done
if [[ -f "hailort.log" ]]; then
    rm hailort.log
fi
