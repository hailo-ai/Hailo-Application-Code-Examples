cmake_minimum_required(VERSION 3.11)
project(vstream_yolov5_yolov7_example_cpp)

set(CMAKE_CXX_STANDARD 20)

set(COMPILE_OPTIONS -Werror -Wall -Wextra -Wconversion -O3 -Wno-reorder -Wno-ignored-qualifiers -Wno-extra -Wno-unused-parameter)

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads)

find_package(HailoRT REQUIRED)

find_package( OpenCV REQUIRED)
message(STATUS "Found OpenCV: " ${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCES
    ./yolov5_yolov7_inference.cpp
    ./common/yolo_hailortpp.cpp
    ./common/yolo_output.cpp
    ./common/yolo_postprocess.cpp
)

link_libraries(stdc++fs)
add_executable(${PROJECT_NAME} ${SOURCES})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNXRUNTIME_INCLUDE_DIR})
include_directories(rapidjson/include)
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS} -fconcepts)
target_link_libraries(${PROJECT_NAME} HailoRT::libhailort ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS})

