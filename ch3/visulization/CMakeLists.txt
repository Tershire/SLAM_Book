cmake_minimum_required(VERSION 3.2)
project(visualization)

set(CMAKE_CXX_FLAGS "-std=c++17 -Wall")

include_directories("/usr/include/eigen3")

# EXTERNAL DEPENDENCIES #
find_package(Pangolin)
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(visualize_cube_pose visualize_cube_pose.cpp)
target_link_libraries(visualize_cube_pose ${Pangolin_LIBRARIES})
