project(Assignment_1)
include_directories(include)

add_executable(${PROJECT_NAME} 
  src/main.cpp)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   "${OpenCV_DIR}/x64/vc15/bin"
                   $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif()


target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} Eigen3::Eigen)