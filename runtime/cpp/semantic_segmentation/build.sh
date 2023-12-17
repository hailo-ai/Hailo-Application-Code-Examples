#!/bin/bash

cmake -H. -Bbuild
cmake --build build

if [[ -f "hailort.log" ]]; then
    rm hailort.log
fi
