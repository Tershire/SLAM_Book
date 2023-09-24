find_package(Pangolin REQUIRED)

add_executable(stereo_reconstruct stereo_reconstruct.cpp)
target_link_libraries(stereo_reconstruct ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
