cmake_minimum_required(VERSION 3.10)
project(OpenCVProject)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(OpenCVProject src/main.cpp)
target_link_libraries(OpenCVProject ${OpenCV_LIBS})