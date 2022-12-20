#!/bin/bash

declare -A COMPILER=( [x86_64]=/usr/bin/gcc )

for ARCH in x86_64
do
    echo "-I- Building ${ARCH}"
    mkdir -p build/${ARCH}
    # HAILORT_VER=4.9.0 cmake -H. -Bbuild/${ARCH} -DARCH=${ARCH} -DCMAKE_C_COMPILER=${COMPILER[${ARCH}]} -DCMAKE_PREFIX_PATH=/local/users/omerw/.local/share/cmake
    HAILORT_VER=4.10.0 cmake -H. -Bbuild/${ARCH} -DARCH=${ARCH} -DCMAKE_C_COMPILER=${COMPILER[${ARCH}]}
    cmake --build build/${ARCH}
done
if [[ -f "hailort.log" ]]; then
    rm hailort.log
fi
