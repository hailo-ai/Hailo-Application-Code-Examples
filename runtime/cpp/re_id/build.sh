#!/bin/bash

RAPIDJSON_DIRECTORY=rapidjson
if [ ! -d "$RAPIDJSON_DIRECTORY" ]; then
  echo "$RAPIDJSON_DIRECTORY does not exist, cloning"
  git clone https://github.com/Tencent/rapidjson
fi

declare -A COMPILER=( [x86_64]=/usr/bin/gcc )

ARCH="x86_64"

echo "-I- Building ${ARCH}"
mkdir -p build/${ARCH}
cmake -H. -Bbuild/${ARCH} -DCMAKE_C_COMPILER=${COMPILER[${ARCH}]}
cmake --build build/${ARCH}

if [[ -f "hailort.log" ]]; then
    rm hailort.log
fi
