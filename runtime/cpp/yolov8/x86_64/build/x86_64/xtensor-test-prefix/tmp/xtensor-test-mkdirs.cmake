# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test"
  "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test-build"
  "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix"
  "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/tmp"
  "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test-stamp"
  "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src"
  "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test-stamp/${subDir}")
endforeach()
