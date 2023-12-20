cmake_minimum_required(VERSION 3.11)
project(scheduler_example)

set(CMAKE_CXX_STANDARD 20)

set(COMPILE_OPTIONS -Wall -Werror -O3)

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
include_directories(${OpenCV_INCLUDE_DIRS})
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})
target_link_libraries(${PROJECT_NAME} HailoRT::libhailort)
target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
