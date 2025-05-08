cmake_minimum_required(VERSION 3.16)
project(dsp_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compilation options
set(COMPILE_OPTIONS 
    -Wall -Wextra -O3 -fconcepts 
    -Wno-ignored-qualifiers 
    -Wno-stringop-truncation 
    -Wno-reorder
)

# Gather sources
file(GLOB SOURCES "./*.cpp")

# Define executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Set compile options
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})

target_link_libraries(${PROJECT_NAME} PRIVATE hailodsp)
