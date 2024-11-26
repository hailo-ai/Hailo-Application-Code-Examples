cmake_minimum_required(VERSION 3.14)
project(clip_async_example_cpp)

set(COMPILE_OPTIONS -Wall -Wextra -Wconversion -O3 -std=c++17 -fconcepts -Wno-ignored-qualifiers -Wno-extra -Wno-stringop-truncation -Wno-reorder)

find_package(Threads)
find_package(HailoRT REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "Found OpenCV: " ${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCES
    ./tokenizer/tokenizer.cpp
    ./tokenizer/cnpy.cpp
    ./tokenizer/run_tokenizer.cpp
    ./clip_example.cpp
)

link_libraries(stdc++fs)
add_executable(${PROJECT_NAME} ${SOURCES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})
target_link_libraries(${PROJECT_NAME} Threads::Threads HailoRT::libhailort ${OpenCV_LIBS})