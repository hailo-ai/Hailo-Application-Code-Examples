cmake_minimum_required(VERSION 3.0.0)

project(infer VERSION 1.0.1 DESCRIPTION "yolov5 inference on hailo-8")

set(COMPILE_OPTIONS -Wall -std=c++14 -Werror -O3)

add_library(infer SHARED
    double_buffer.hpp
    common.h
    yolo_post_processing.hpp
    yolo_post_processing.cpp
    infer.cpp)

set_target_properties(infer PROPERTIES PUBLIC_HEADER include/infer.hpp)

install(TARGETS infer
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

find_package(Threads REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(HailoRT REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(infer PRIVATE HailoRT::libhailort Threads::Threads ${OpenCV_LIBS})

if(WIN32)
    target_compile_options(infer PRIVATE
        /DWIN32_LEAN_AND_MEAN
        /DNOMINMAX                  # NOMINMAX is required in order to play nice with std::min/std::max (otherwise Windows.h defines it's own)
        /wd4201 /wd4251
    )
endif()

set_target_properties(infer PROPERTIES CXX_STANDARD 14)
