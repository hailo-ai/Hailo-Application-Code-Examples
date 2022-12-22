cmake_minimum_required(VERSION 3.0.0)
project(vstream_re_id_example)

set(CMAKE_CXX_STANDARD 20)

set(COMPILE_OPTIONS -Wall -Werror -O3 -DNDEBUG -fconcepts -Wno-unused-local-typedefs)

find_package(Threads)
find_package(HailoRT 4.10.0 EXACT REQUIRED)
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV: " ${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCES
    ./*.hpp
    ./*.cpp
)

link_libraries(stdc++fs)
add_executable(${PROJECT_NAME} ${SOURCES})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("./xtl/include" "./xtensor/include" "./common/" "./rapidjson/include/rapidjson")
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} HailoRT::libhailort Threads::Threads)
