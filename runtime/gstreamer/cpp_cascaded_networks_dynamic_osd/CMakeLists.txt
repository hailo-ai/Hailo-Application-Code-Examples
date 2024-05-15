# This CMakeLists.txt may only run after sourcing the cross develeopment toolchain

cmake_minimum_required(VERSION 3.11)
project(cascaded_networks_dynamic_osd)

set(LIB_HAILORT $ENV{SDKTARGETSYSROOT}/usr/lib/libhailort.so)
set(LIB_HAILO_MEDIALIB
    $ENV{SDKTARGETSYSROOT}/usr/lib/libhailo_media_library_common.so.0
    $ENV{SDKTARGETSYSROOT}/usr/lib/libhailo_media_library_frontend.so.0
    $ENV{SDKTARGETSYSROOT}/usr/lib/libhailo_media_library_encoder.so.0
    $ENV{SDKTARGETSYSROOT}/usr/lib/libgstmedialibutils.so
)
set(HAILO_INCLUDE_DIRS 
    $ENV{SDKTARGETSYSROOT}/usr/include/hailo/    
    $ENV{SDKTARGETSYSROOT}/usr/include/hailo/media_library/
    $ENV{SDKTARGETSYSROOT}/usr/include/hailort/
    $ENV{SDKTARGETSYSROOT}/usr/include/gstreamer-1.0/
    $ENV{SDKTARGETSYSROOT}/usr/lib/glib-2.0/include/gst/
	$ENV{SDKTARGETSYSROOT}/usr/include/glib-2.0/
    $ENV{SDKTARGETSYSROOT}/usr/include/
    $ENV{SDKTARGETSYSROOT}/usr/lib/glib-2.0/include/        
)

set (GSTREAMER_LINK_LIBS
        $ENV{SDKTARGETSYSROOT}/usr/lib/libgstreamer-1.0.so
	    $ENV{SDKTARGETSYSROOT}/usr/lib/libglib-2.0.so.0
        $ENV{SDKTARGETSYSROOT}/usr/lib/libgobject-2.0.so.0
)

set(USR_OPENCV_INCLUDE_DIRS $ENV{SDKTARGETSYSROOT}/usr/include/opencv4)
set(USR_PP_INCLUDE_DIRS 
    $ENV{SDKTARGETSYSROOT}/usr/src/debug/hailo-post-processes/3.27.2-r0/git/core/hailo/libs/postprocesses/
    $ENV{SDKTARGETSYSROOT}/usr/src/debug/hailo-post-processes/3.27.2-r0/git/core/hailo/libs/postprocesses/common/
    $ENV{SDKTARGETSYSROOT}/usr/src/debug/hailo-post-processes/3.27.2-r0/git/core/hailo/libs/postprocesses/detection/
    $ENV{SDKTARGETSYSROOT}/usr/src/debug/hailo-post-processes/3.27.2-r0/git/core/hailo/general/    
    $ENV{SDKTARGETSYSROOT}/usr/include/glib-2.0/gobject/
)
set(OPENCV_VIDEOIO_LIBS 
	$ENV{SDKTARGETSYSROOT}/usr/lib/libopencv_videoio.so.4.5.5  
	$ENV{SDKTARGETSYSROOT}/usr/lib/libopencv_imgproc.so.4.5.5 
	$ENV{SDKTARGETSYSROOT}/usr/lib/libopencv_core.so.405 
	$ENV{SDKTARGETSYSROOT}/usr/lib/libopencv_imgcodecs.so.405
) 

find_package(Threads)
file(GLOB SOURCES
    ./cascaded_networks_dynamic_osd.cpp
    ./yolo_hailortpp.hpp
    ./yolo_hailortpp.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})
include_directories(${HAILO_INCLUDE_DIRS} ${USR_PP_INCLUDE_DIRS} ${USR_OPENCV_INCLUDE_DIRS})
add_library(yolo_hailortpp SHARED ./yolo_hailortpp.cpp)
target_compile_options(${PROJECT_NAME} PRIVATE ${CXXFLAGS})
target_link_libraries(${PROJECT_NAME} Threads::Threads ${LIB_HAILORT} ${OpenCV_LIBS} ${OPENCV_VIDEOIO_LIBS} ${GSTREAMER_LINK_LIBS} ${LIB_HAILO_MEDIALIB})
