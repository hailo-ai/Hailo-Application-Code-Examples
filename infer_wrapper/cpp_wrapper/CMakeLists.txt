cmake_minimum_required(VERSION 3.0.0)

project(infer)

set(COMPILE_OPTIONS -Wall -std=c++14 -Werror -O3)

find_package(Threads REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(HailoRT 4.14.0 EXACT REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(infer infer.cpp)
target_link_libraries(infer PRIVATE HailoRT::libhailort Threads::Threads ${OpenCV_LIBS})

if(WIN32)
    target_compile_options(infer PRIVATE
        /DWIN32_LEAN_AND_MEAN
        /DNOMINMAX                  # NOMINMAX is required in order to play nice with std::min/std::max (otherwise Windows.h defines it's own)
        /wd4201 /wd4251
    )
endif()

set_target_properties(infer PROPERTIES CXX_STANDARD 14)