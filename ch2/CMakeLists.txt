cmake_minimum_required( VERSION 3.22 )
project( hello_SLAM )

add_executable( hello_SLAM hello_SLAM.cpp )
add_executable( hello_SLAM_using_lib hello_SLAM_using_lib.cpp )

add_library( hello_static        lib_hello_SLAM.cpp )
add_library( hello_shared SHARED lib_hello_SLAM.cpp )

target_link_libraries( hello_SLAM_using_lib hello_shared )
