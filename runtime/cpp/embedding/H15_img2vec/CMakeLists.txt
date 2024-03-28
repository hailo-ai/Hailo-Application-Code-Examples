# This CMakeLists.txt may only run after sourcing the cross develeopment toolchain

cmake_minimum_required(VERSION 3.11)
project(feature_extractor_cross_compilation_h15)

set(LIB_HAILORT $ENV{SDKTARGETSYSROOT}/usr/lib/libhailort.so)
set(HAILO_INCLUDE_DIRS $ENV{SDKTARGETSYSROOT}/usr/include/hailort)
set(USR_OPENCV_INCLUDE_DIRS $ENV{SDKTARGETSYSROOT}/usr/include/opencv4)
set(OPENCV_VIDEOIO_LIBS 
	$ENV{SDKTARGETSYSROOT}/usr/lib/libopencv_videoio.so.4.5.5  
	$ENV{SDKTARGETSYSROOT}/usr/lib/libopencv_imgproc.so.4.5.5 
	$ENV{SDKTARGETSYSROOT}/usr/lib/libopencv_core.so.405 
	$ENV{SDKTARGETSYSROOT}/usr/lib/libopencv_imgcodecs.so.405
) 

find_package(Threads)
file(GLOB SOURCES
    postprocess.cpp
)

add_library(${PROJECT_NAME} SHARED ${SOURCES})
include_directories(${HAILO_INCLUDE_DIRS} ${USR_OPENCV_INCLUDE_DIRS})
target_compile_options(${PROJECT_NAME} PRIVATE ${CXXFLAGS})
target_link_libraries(${PROJECT_NAME} Threads::Threads ${LIB_HAILORT} ${OpenCV_LIBS} ${OPENCV_VIDEOIO_LIBS})
