project(image_basics)

add_executable(basic_image_operations basic_image_operations.cpp)
target_link_libraries(basic_image_operations ${OpenCV_LIBS})

add_executable(undistort undistort.cpp)
target_link_libraries(undistort ${OpenCV_LIBS})
