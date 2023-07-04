add_executable(run_kitti_stereo run_kitti_stereo.cpp)

target_link_libraries(run_kitti_stereo my_VO ${THIRD_PARTY_LIBS})