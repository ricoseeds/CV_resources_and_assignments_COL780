project( assignment_minus_1 )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_STANDARD} -O3 -std=c++11")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   "${OpenCV_DIR}/x64/vc12/bin"
                   $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(include/)
FILE(GLOB SRCFILES src/*.cpp src/*.c)

add_executable(${PROJECT_NAME} ${SRCFILES} )
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "${PROJECT_NAME}.out")
set(EXECUTABLE_OUTPUT_PATH "bin")
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
