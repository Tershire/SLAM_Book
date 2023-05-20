cmake_minimum_required(VERSION 3.2)
project(VO_1)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV 4 REQUIRED)
# find_package(G2O REQUIRED)
# find_package(Sophus REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        # ${G2O_INCLUDE_DIRS}
        # ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3/"
)

add_executable(orb_1to2 orb_1to2.cpp)
target_link_libraries(orb_1to2 ${OpenCV_LIBS})

add_executable(orb_scratch orb_scratch.cpp)
target_link_libraries(orb_scratch ${OpenCV_LIBS})
