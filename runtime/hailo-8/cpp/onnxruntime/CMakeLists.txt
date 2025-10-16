cmake_minimum_required(VERSION 3.5.0)
project(hailo_ort_example)

file(GLOB ALL_SUBDIRS "/opt/*")
set(ONNXRUNTIME_DIR "")

foreach(DIR ${ALL_SUBDIRS})
    if(IS_DIRECTORY ${DIR} AND "${DIR}" MATCHES ".*onnxruntime.*")
        set(ONNXRUNTIME_DIR ${DIR})
        break()
    endif()
endforeach()

if(ONNXRUNTIME_DIR)
    message(STATUS "Found directory: ${ONNXRUNTIME_DIR}")
    find_library(ONNXRUNTIME onnxruntime ${ONNXRUNTIME_DIR}/lib)
    set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_DIR}/include)
else()
    message(STATUS "No directory containing 'onnxruntime' was found.")
endif()

set(COMPILE_OPTIONS -Wall -Werror -g -O3)

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
    ./*.cpp
)

link_libraries(stdc++fs)
add_executable(${PROJECT_NAME} ${SOURCES})
include_directories(${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIR})
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})
target_link_libraries(${PROJECT_NAME} HailoRT::libhailort ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS} ${ONNXRUNTIME})
