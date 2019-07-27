cmake_minimum_required(VERSION 3.2)
project( COL780 )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_subdirectory(DisplayImage)

add_subdirectory(HistEqualization)


