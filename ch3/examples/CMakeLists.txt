cmake_minimum_required(VERSION 3.22)
project(examples)

# HEADER FILE /////////////////////////////////////////////////////////////////
include_directories("/usr/include/eigen3")

# PACKAGE /////////////////////////////////////////////////////////////////////
# set(Pangolin_DIR "/home/tershire/SLAM_book/ext_libs/Pangolin")
find_package(Pangolin REQUIRED)

# EXECUTABLE //////////////////////////////////////////////////////////////////
add_executable(transform_bots transform_bots.cpp)
add_executable(plot_trajectory plot_trajectory.cpp)

# LINK ////////////////////////////////////////////////////////////////////////
target_link_libraries(plot_trajectory ${Pangolin_LIBRARIES})
