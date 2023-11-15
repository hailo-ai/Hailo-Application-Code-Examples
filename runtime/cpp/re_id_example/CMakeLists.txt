cmake_minimum_required(VERSION 3.0.0)
project(vstream_re_id_example)

set(CMAKE_CXX_STANDARD 20)

set(COMPILE_OPTIONS -Wall -Werror -O3 -DNDEBUG -fconcepts -Wno-unused-local-typedefs)

set(SYMLINK_PATH "/usr/lib/libhailort.so")

# Run the readlink command to get the target of the symbolic link
execute_process(
    COMMAND readlink -f ${SYMLINK_PATH}
    OUTPUT_VARIABLE SYMLINK_TARGET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the version of the libhailort
string(LENGTH ${SYMLINK_TARGET} TARGET_LENGTH)
math(EXPR START_INDEX "${TARGET_LENGTH} - 6")
string(SUBSTRING ${SYMLINK_TARGET} ${START_INDEX} 6 HAILORT_VERSION)

message("HailoRT lib version is: ${HAILORT_VERSION}")


find_package(Threads)
find_package(HailoRT ${HAILORT_VERSION} EXACT REQUIRED)
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV: " ${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCES
    ./*.cpp
    ./common/*.cpp
)

include(ExternalProject)

set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

ExternalProject_Add(xtl-test
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
)

ExternalProject_Add(xtensor-test
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -Dxtl_DIR=${BASE_DIR}/build/x86_64/external/share/cmake/xtl/
)

include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)

link_libraries(stdc++fs)
add_executable(${PROJECT_NAME} ${SOURCES})
add_dependencies(${PROJECT_NAME} xtl-test xtensor-test)
include_directories(${OpenCV_INCLUDE_DIRS} rapidjson/include)
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} HailoRT::libhailort Threads::Threads)
