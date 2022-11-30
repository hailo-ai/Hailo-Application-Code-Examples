cmake_minimum_required(VERSION 3.0.0)
project(vstream_re_id_example)

set(HAILORT_VER $ENV{HAILORT_VER})
set(HAILORT_LIB /usr/lib/libhailort.so.${HAILORT_VER})
set(CMAKE_CXX_STANDARD 20)

# set(COMPILE_OPTIONS -Wall -Werror -O3 -DEBUG -fconcepts)
#set(COMPILE_OPTIONS -Wall -Werror -fconcepts -g -O0)
set(COMPILE_OPTIONS -Wall -fconcepts -fpermissive -g -O0)

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads)

find_package( OpenCV REQUIRED)
message(STATUS "Found OpenCV: " ${OpenCV_INCLUDE_DIRS})

set(XTENSOR_INCLUDE_DIR /opt/include/)

file(GLOB SOURCES
    ./*.hpp
    ./*.cpp
)

link_libraries(stdc++fs)
add_executable(${PROJECT_NAME}.${HAILORT_VER} ${SOURCES})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${XTENSOR_INCLUDE_DIR})
target_compile_options(${PROJECT_NAME}.${HAILORT_VER} PRIVATE ${COMPILE_OPTIONS})
target_link_libraries(${PROJECT_NAME}.${HAILORT_VER} ${HAILORT_LIB})
target_link_libraries(${PROJECT_NAME}.${HAILORT_VER} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${PROJECT_NAME}.${HAILORT_VER} ${OpenCV_LIBS})

