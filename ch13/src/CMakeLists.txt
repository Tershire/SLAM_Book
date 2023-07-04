add_library(my_VO SHARED
            frame.cpp
            map_point.cpp
            map.cpp
            camera.cpp
            config.cpp
            feature.cpp
            frontend.cpp
            backend.cpp
            viewer.cpp
            visual_odometry.cpp
            dataset.cpp)

target_link_libraries(my_VO
                      ${THIRD_PARTY_LIBS})