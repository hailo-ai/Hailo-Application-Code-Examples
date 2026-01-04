
cmake_minimum_required(VERSION 3.12)
project(onnxrt_hailo_pipeline CXX)

# ----- Compiler check (GNU only) -----
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11")
        message(FATAL_ERROR
            "GCC 11+ required. Found version ${CMAKE_CXX_COMPILER_VERSION} at ${CMAKE_CXX_COMPILER}\n"
            "Fix:\n"
            "  • Install GCC 11+: sudo apt install gcc-11 g++-11\n"
            "  • Reconfigure: cmake -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11 <src>\n"
            "  • Delete your build directory and rerun cmake.")
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# ---- ONNX Runtime under /opt ----
file(GLOB ALL_SUBDIRS "/opt/*")
set(ONNXRUNTIME_DIR "")
foreach(DIR ${ALL_SUBDIRS})
    if (IS_DIRECTORY ${DIR} AND "${DIR}" MATCHES ".*onnxruntime.*")
        set(ONNXRUNTIME_DIR ${DIR})
        break()
    endif()
endforeach()
if (ONNXRUNTIME_DIR)
    message(STATUS "Found directory: ${ONNXRUNTIME_DIR}")
    find_library(ONNXRUNTIME onnxruntime HINTS ${ONNXRUNTIME_DIR}/lib REQUIRED)
    set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_DIR}/include)
else()
    message(FATAL_ERROR "No directory containing 'onnxruntime' was found in /opt/")
endif()

# ---- ExternalProject for xtl/xtensor (header-only) ----
include(ExternalProject)

# A single install prefix where headers will land:
set(EXTERNAL_PREFIX ${CMAKE_BINARY_DIR}/external)

ExternalProject_Add(xtl-test
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl
    GIT_TAG 0.7.7                   # pin a known version (recommended)
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PREFIX}
    INSTALL_DIR ${EXTERNAL_PREFIX}
)

ExternalProject_Add(xtensor-test
    DEPENDS xtl-test                # ensure xtl is ready first
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor
    GIT_TAG 0.24.6                  # pin a known version (recommended)
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PREFIX}
        -Dxtl_DIR=${EXTERNAL_PREFIX}/share/cmake/xtl
    INSTALL_DIR ${EXTERNAL_PREFIX}
)

# xtensor include path (after external install)
set(XTENSOR_INCLUDE_DIR ${EXTERNAL_PREFIX}/include)

# ---- Your target ----
add_executable(onnxrt_hailo_pipeline
    onnxrt_hailo_pipeline.cpp
    onnx_decode.cpp
    instance_seg_postprocess.cpp
    ../common/toolbox.cpp
    ../common/hailo_infer.cpp
)

# Ensure the external projects run before building the executable sources
add_dependencies(onnxrt_hailo_pipeline xtl-test xtensor-test)

target_include_directories(onnxrt_hailo_pipeline PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${XTENSOR_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_link_libraries(onnxrt_hailo_pipeline
    PRIVATE
        ${OpenCV_LIBS}
        Threads::Threads
        ${ONNXRUNTIME}
        hailort
)

# Per-config compile flags (portable)
target_compile_options(onnxrt_hailo_pipeline PRIVATE
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
    $<$<CONFIG:RelWithDebInfo>:-O2 -g -DNDEBUG>
    $<$<CONFIG:Debug>:-O0 -g>
)
