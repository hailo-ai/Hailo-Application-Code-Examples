#!/bin/bash

# echo "-I- Building ${ARCH} for compiler ${COMPILER}"
mkdir -p build #/${ARCH}
cmake -H. -Bbuild #/${ARCH}
# cmake -H. -Bbuild -DCMAKE_BUILD_TYPE=Debug
cmake --build build #/${ARCH}

# if [[ -f "hailort.log" ]]; then
#     rm hailort.log
# fi
