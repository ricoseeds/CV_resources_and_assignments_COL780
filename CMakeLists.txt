cmake_minimum_required(VERSION 3.2)
project( COL780 )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(vendor/)

add_subdirectory(Assignments/DisplayImage)
add_subdirectory(Assignments/assignment_minus_1)
add_subdirectory(Assignments/HistEqualization)




