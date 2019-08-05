
include_directories(include)

add_executable(EigenExample 
  src/EigenExample.cpp
  include/EigenExample.h)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
add_custom_command(TARGET EigenExample POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   "${OpenCV_DIR}/x64/vc15/bin"
                   $<TARGET_FILE_DIR:EigenExample>)
endif()


target_link_libraries( EigenExample ${OpenCV_LIBS} Eigen3::Eigen)