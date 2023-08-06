cmake_minimum_required(VERSION 3.11)
project(vstream_yolov8_example_cpp)
project(yolov8_cross_compilation_h15)

set(LIB_HAILORT /opt/poky/4.0.2/sysroots/armv8a-poky-linux/usr/lib/libhailort.so)
set(HAILO_INCLUDE_DIRS /opt/poky/4.0.2/sysroots/armv8a-poky-linux/usr/include/hailort)
set(USR_OPENCV_INCLUDE_DIRS /opt/poky/4.0.2/sysroots/armv8a-poky-linux/usr/include/opencv4)
# set(OPENCV_VIDEOIO_LIB /opt/poky/4.0.2/sysroots/armv8a-poky-linux/usr/lib/libopencv_videoio.so)
set(OPENCV_VIDEOIO_LIBS /opt/poky/4.0.2/sysroots/armv8a-poky-linux/usr/lib/libopencv_videoio.so.4.5.5 /opt/poky/4.0.2/sysroots/armv8a-poky-linux/usr/lib/libopencv_imgproc.so.4.5.5 /opt/poky/4.0.2/sysroots/armv8a-poky-linux/usr/lib/libopencv_core.so.405 /opt/poky/4.0.2/sysroots/armv8a-poky-linux/usr/lib/libopencv_imgcodecs.so.405) 


# set(COMPILE_OPTIONS -Wall -Wextra -Wconversion -O3 -DNDEBUG -Wno-conversion -Wno-unused-but-set-parameter -Wno-extra -Wno-reorder -Wno-unused-local-typedefs)
set(COMPILE_OPTIONS -pipe -g -feliminate-unused-debug-types -O3)

find_package(Threads)
# find_package(HailoRT REQUIRED)
# find_package(OpenCV REQUIRED)

# message(STATUS "Found OpenCV: " ${OpenCV_INCLUDE_DIRS})

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
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -Dxtl_DIR=${CMAKE_CURRENT_SOURCE_DIR}/build/aarch64/external/share/cmake/xtl/
)

include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)

# message("CMAKE_CXX_FLAGS_DEFAULT is ${CMAKE_CXX_FLAGS}")

# link_libraries(stdc++fs)
add_executable(${PROJECT_NAME} ${SOURCES})
add_dependencies(${PROJECT_NAME} xtl-test xtensor-test)
# include_directories(${HAILO_INCLUDE_DIRS} ${USR_OPENCV_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
include_directories(${HAILO_INCLUDE_DIRS} ${USR_OPENCV_INCLUDE_DIRS})
# target_include_directories(${PROJECT_NAME} PUBLIC ${HAILO_INCLUDE_DIR} ${USR_OPENCV_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
# target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS} -lopencv_imgproc)
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})
target_link_libraries(${PROJECT_NAME} Threads::Threads ${LIB_HAILORT} ${OpenCV_LIBS} ${OPENCV_VIDEOIO_LIBS})
