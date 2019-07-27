
add_executable( HistEqualization HistEqualization.cpp )

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
add_custom_command(TARGET HistEqualization POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   "${OpenCV_DIR}/x64/vc12/bin"
                   $<TARGET_FILE_DIR:HistEqualization>)
endif()

target_link_libraries( HistEqualization ${OpenCV_LIBS} )