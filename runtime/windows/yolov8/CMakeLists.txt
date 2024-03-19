cmake_minimum_required(VERSION 3.11)
project(yolov8_example)

set(CMAKE_CXX_STANDARD 20)

set(COMPILE_OPTIONS -Wall)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ") 

find_package(Threads REQUIRED)
find_package(HailoRT REQUIRED)
find_package(OpenCV  REQUIRED)

file(GLOB SOURCES ./*.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})
target_link_libraries(${PROJECT_NAME} HailoRT::libhailort)
target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})