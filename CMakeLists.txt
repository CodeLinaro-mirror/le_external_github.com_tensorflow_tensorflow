cmake_minimum_required(VERSION 3.3.1)
project(tensorflow)

option(TFLITE_WITH_RUY "Enable RUY" ON)
if(TFLITE_WITH_RUY)
    add_definitions(-DTFLITE_WITH_RUY)
endif(TFLITE_WITH_RUY)

set(TFLITE_TOP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TFLITE_TOP_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -fno-short-enums -Ofast -Wl,-no-undefined -lstdc++")

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
include_directories(tensorflow/lite/tools/make/downloads/absl)

install(
    DIRECTORY tensorflow/core
    DESTINATION include/tflite/tensorflow
    FILES_MATCHING
    PATTERN "*.h"
)

add_subdirectory(tensorflow/lite)
add_subdirectory(tensorflow/core)
