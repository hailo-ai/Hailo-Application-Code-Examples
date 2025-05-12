cmake_minimum_required(VERSION 3.10.0)
project(depth_estimation_example_cpp)

find_package(Threads)
find_package(OpenCV REQUIRED)
find_package(HailoRT REQUIRED)

set(COMPILE_OPTIONS_CPP -Wall -Werror -O3)

file(GLOB SOURCES ./*.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})

include_directories(/usr/include ./)
include_directories( ${OpenCV_INCLUDE_DIRS})

target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS_CPP})
target_link_libraries(${PROJECT_NAME} HailoRT::libhailort Threads::Threads)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

