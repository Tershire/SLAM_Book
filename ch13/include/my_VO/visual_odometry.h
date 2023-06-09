// visual_odometry.h
// 2023 JUL 03
// Tershire reformatted the code...

// Created by gaoxiang.

#pragma once
#ifndef MY_VO_VISUAL_ODOMETRY_H
#define MY_VO_VISUAL_ODOMETRY_H

#include "my_VO/backend.h"
#include "my_VO/common_include.h"
#include "my_VO/dataset.h"
#include "my_VO/frontend.h"
#include "my_VO/viewer.h"

namespace my_VO
{

/**
 * VO 对外接口
 */
class VisualOdometry
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<VisualOdometry> Ptr;

    /// constructor with config file
    VisualOdometry(std::string &config_path);

    /**
     * do initialization things before run
     * @return true if success
     */
    bool Init();

    /**
     * start vo in the dataset
     */
    void Run();

    /**
     * Make a step forward in dataset
     */
    bool Step();

    /// 获取前端状态
    FrontendStatus GetFrontendStatus() const {return frontend_->GetStatus();}

   private:
    bool inited_ = false;
    std::string config_file_path_;

    Frontend::Ptr frontend_ = nullptr;
    Backend::Ptr backend_ = nullptr;
    Map::Ptr map_ = nullptr;
    Viewer::Ptr viewer_ = nullptr;

    // dataset
    Dataset::Ptr dataset_ = nullptr;
};

}  // namespace my_VO

#endif  // MY_VO_VISUAL_ODOMETRY_H
