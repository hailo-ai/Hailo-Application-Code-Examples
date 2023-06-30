#!/bin/bash

RAPIDJSON_DIRECTORY=rapidjson
if [ ! -d "$RAPIDJSON_DIRECTORY" ]; then
  echo "$RAPIDJSON_DIRECTORY does not exist, cloning"
  git clone https://github.com/Tencent/rapidjson
fi

declare -A COMPILER=( [x86_64]=/usr/bin/gcc
                      [aarch64]=/usr/bin/aarch64-linux-gnu-gcc
                      [armv7l]=/usr/bin/arm-linux-gnueabi-gcc )

for ARCH in x86_64 # aarch64
do
    echo "-I- Building ${ARCH}"
    mkdir -p build/${ARCH}
    HAILORT_VER=4.10.0 cmake -H. -Bbuild/${ARCH} -DARCH=${ARCH} -DCMAKE_C_COMPILER=${COMPILER[${ARCH}]}
    cmake --build build/${ARCH}
done

if [[ -f "hailort.log" ]]; then
    rm hailort.log
fi
