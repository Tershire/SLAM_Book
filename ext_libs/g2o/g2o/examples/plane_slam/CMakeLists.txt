add_executable(simulator_3d_plane
  simulator_3d_plane.cpp
)
target_link_libraries(simulator_3d_plane solver_eigen types_slam3d_addons)
set_target_properties(simulator_3d_plane PROPERTIES OUTPUT_NAME simulator_3d_plane)
