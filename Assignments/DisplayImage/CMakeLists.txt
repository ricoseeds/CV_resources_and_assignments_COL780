
add_executable( DisplayImage DisplayImage.cpp )

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
add_custom_command(TARGET DisplayImage POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   "${OpenCV_DIR}/x64/vc15/bin"
                   $<TARGET_FILE_DIR:DisplayImage>)
endif()

target_link_libraries( DisplayImage ${OpenCV_LIBS} )