if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
add_custom_command(TARGET assignment_minus_1 POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   "${OpenCV_DIR}/x64/vc12/bin"
                   $<TARGET_FILE_DIR:assignment_minus_1>)
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_STANDARD} -O3 -std=c++11")
include_directories(include/)
FILE(GLOB SRCFILES src/*.cpp src/*.c)
add_executable(assignment_minus_1 ${SRCFILES} )
target_link_libraries( assignment_minus_1 ${OpenCV_LIBS} )
