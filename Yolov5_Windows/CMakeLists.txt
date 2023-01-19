cmake_minimum_required(VERSION 3.11)
project(cpp_another_vstreams_example)

set(CMAKE_CXX_STANDARD 20)

set(COMPILE_OPTIONS -Wall)

# /wd5045 removes Spectre mitigation warning
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MT /wd5045") 

set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0) 
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Threads)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIR} )

message(STATUS "Found OpenCV: " ${OpenCV_INCLUDE_DIRS})

find_package( HailoRT 4.12.0 REQUIRED )

file(GLOB SOURCES
    ./*.hpp
    ./*.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNXRUNTIME_INCLUDE_DIR})
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})
target_link_libraries(${PROJECT_NAME} HailoRT::libhailort)
target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
