
add_executable( DisplayImage DisplayImage.cpp )

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
add_custom_command(TARGET DisplayImage POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   "${OpenCV_DIR}/x64/vc12/bin"
                   $<TARGET_FILE_DIR:DisplayImage>)
endif()

target_link_libraries( DisplayImage ${OpenCV_LIBS} )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_STANDARD} -O3 -std=c++11")