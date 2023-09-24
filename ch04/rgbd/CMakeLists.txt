find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

add_definitions("-D SOPHUS_USE_BASIC_LOGGING")

find_package(Pangolin REQUIRED)

add_executable(rgbd_reconstruct rgbd_reconstruct.cpp)
target_link_libraries(rgbd_reconstruct ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
