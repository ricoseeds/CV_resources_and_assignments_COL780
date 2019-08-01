cmake_minimum_required(VERSION 3.2)
project( COL780 )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_subdirectory(Assignments/DisplayImage)
add_subdirectory(Assignments/HistEqualization)
add_subdirectory(Assignments/EigenExample)



