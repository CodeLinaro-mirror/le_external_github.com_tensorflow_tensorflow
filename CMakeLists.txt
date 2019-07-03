cmake_minimum_required(VERSION 3.3.1)
project(tensorflow)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -fno-short-enums -Ofast -Wl,-no-undefined -lstdc++")

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_subdirectory(tensorflow/lite)