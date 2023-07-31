#!/bin/bash

ARCH=aarch64
COMPILER=/usr/bin/aarch64-linux-gnu-gcc

echo "-I- Building ${ARCH} for compiler ${COMPILER}"
mkdir -p build/${ARCH}
cmake -H. -Bbuild/${ARCH}
cmake --build build/${ARCH}

if [[ -f "hailort.log" ]]; then
    rm hailort.log
fi
