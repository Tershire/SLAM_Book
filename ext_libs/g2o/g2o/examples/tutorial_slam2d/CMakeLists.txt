add_library(tutorial_slam2d_library ${G2O_LIB_TYPE}
  edge_se2_pointxy.cpp  se2.h                vertex_point_xy.h
  edge_se2.cpp    edge_se2_pointxy.h    vertex_se2.cpp
  edge_se2.h      vertex_point_xy.cpp  vertex_se2.h
  parameter_se2_offset.h parameter_se2_offset.cpp
  types_tutorial_slam2d.h types_tutorial_slam2d.cpp
  simulator.h simulator.cpp
  g2o_tutorial_slam2d_api.h
)

set_target_properties(tutorial_slam2d_library PROPERTIES OUTPUT_NAME ${LIB_PREFIX}tutorial_slam2d)
set_target_properties(tutorial_slam2d_library PROPERTIES
  VERSION ${G2O_LIB_VERSION}
  SOVERSION ${G2O_LIB_SOVERSION})

target_link_libraries(tutorial_slam2d_library core solver_eigen)

add_executable(tutorial_slam2d
  tutorial_slam2d.cpp
)

set_target_properties(tutorial_slam2d PROPERTIES OUTPUT_NAME tutorial_slam2d)

target_link_libraries(tutorial_slam2d tutorial_slam2d_library)
