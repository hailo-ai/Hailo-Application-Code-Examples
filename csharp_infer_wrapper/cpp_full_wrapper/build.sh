#!/bin/bash

cmake -H. -Bbuild -DCMAKE_BUILD_TYPE=Release
cmake --build build --config release
