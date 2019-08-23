cmake_minimum_required(VERSION 3.2)
project( COL780 )

find_package( OpenCV 4.1 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ext/ )

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_STANDARD} -O3 -std=c++11")
add_subdirectory(Assignments/Assignment_1)



