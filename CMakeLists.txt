cmake_minimum_required(VERSION 2.8.4)
project(OpenCVTest)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package( OpenCV REQUIRED )
set(SOURCE_FILES main.cpp)
add_executable(OpenCVTest ${SOURCE_FILES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( OpenCVTest ${OpenCV_LIBS} )