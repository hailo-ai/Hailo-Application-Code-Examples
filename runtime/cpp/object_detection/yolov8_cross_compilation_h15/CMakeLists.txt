# This CMakeLists.txt may only run after sourcing the cross develeopment toolchain

cmake_minimum_required(VERSION 3.11)
project(yolov8_cross_compilation_h15)

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
add_executable(${PROJECT_NAME} ${SOURCES})
add_dependencies(${PROJECT_NAME} xtl-test xtensor-test)

include_directories(${HAILO_INCLUDE_DIRS} ${USR_OPENCV_INCLUDE_DIRS})
target_compile_options(${PROJECT_NAME} PRIVATE ${CXXFLAGS})
target_link_libraries(${PROJECT_NAME} Threads::Threads ${LIB_HAILORT} ${OpenCV_LIBS} ${OPENCV_VIDEOIO_LIBS})
