cmake_minimum_required(VERSION 3.11)
project(vstream_yolov8_example_cpp)

set(BASE_DIR ~/yolov8/Hailo-Application-Code-Examples/yolov8_cpp)
set(USR_OPENCV_INCLUDE_DIR /usr/include/opencv4/)
set(OPENCV_VIDEOIO_LIB /usr/lib/aarch64-linux-gnu/libopencv_videoio.so)

set(COMPILE_OPTIONS -Wall -Wextra -Wconversion -O3 -DNDEBUG -Wno-conversion -Wno-unused-but-set-parameter -Wno-extra -Wno-reorder -Wno-unused-local-typedefs)

find_package(Threads)
find_package(HailoRT REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "Found OpenCV: " ${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCES
    yolov8_inference.cpp
    yolov8_postprocess.cpp
)

include(ExternalProject)

set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

ExternalProject_Add(xtl-test
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
)

ExternalProject_Add(xtensor-test
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -Dxtl_DIR=${BASE_DIR}/build/aarch64/external/share/cmake/xtl/
)

include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)

link_libraries(stdc++fs)
add_executable(${PROJECT_NAME} ${SOURCES})
add_dependencies(${PROJECT_NAME} xtl-test xtensor-test)
include_directories(${USR_OPENCV_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})
target_link_libraries(${PROJECT_NAME} Threads::Threads HailoRT::libhailort ${OpenCV_LIBS} ${OPENCV_VIDEOIO_LIB})
