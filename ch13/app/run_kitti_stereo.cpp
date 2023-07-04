// run_kitti_stereo.cpp
// 2023 JUL 03
// Tershire reformatted the code...

// Created by gaoxiang on 19-5-4.

#include <gflags/gflags.h>
#include "my_VO/visual_odometry.h"

DEFINE_string(config_file, "./config/default.yaml", "config file path");

int main(int argc, char **argv)
{
    google::ParseCommandLineFlags(&argc, &argv, true);

    my_VO::VisualOdometry::Ptr vo(
        new my_VO::VisualOdometry(FLAGS_config_file));
    assert(vo->Init() == true);
    vo->Run();

    return 0;
}
