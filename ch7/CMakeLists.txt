cmake_minimum_required(VERSION 3.2)
project(VO_1)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
# set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")

# C++17 needed for this version of g2o used 
set(CMAKE_CXX_FLAGS "-std=c++17 -O2 ${SSE_FLAGS} -msse4")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV 4 REQUIRED)
find_package(G2O REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3/"
)

# needed. otherwise fmt error (ex. /usr/include/fmt/ostream.h:72:23: 
#                              error: ‘formatbuf’ was not declared in this scope; 
#                              did you mean ‘format_buf’?)
add_definitions("-D SOPHUS_USE_BASIC_LOGGING")

add_executable(orb_1to2 orb_1to2.cpp)
add_executable(orb_scratch orb_scratch.cpp)
add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
add_executable(triangulation triangulation.cpp)
add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)

target_link_libraries(orb_1to2 ${OpenCV_LIBS})
target_link_libraries(orb_scratch ${OpenCV_LIBS})
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})
target_link_libraries(triangulation ${OpenCV_LIBS})
target_link_libraries(pose_estimation_3d2d ${OpenCV_LIBS} 
                                           g2o_core g2o_stuff)