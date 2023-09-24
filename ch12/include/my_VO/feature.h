// dataset.h
// 2023 JUL 03
// Tershire reformatted the code...

// Created by gaoxiang.

#pragma once

#ifndef MY_VO_FEATURE_H
#define MY_VO_FEATURE_H

#include <memory>
#include <opencv2/features2d.hpp>
#include "my_VO/common_include.h"

namespace my_VO
{

struct Frame;
struct MapPoint;

/**
 * 2D 特征点
 * 在三角化之后会被关联一个地图点
 */
struct Feature
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Feature> Ptr;

    std::weak_ptr<Frame> frame_;         // 持有该feature的frame
    cv::KeyPoint position_;              // 2D提取位置
    std::weak_ptr<MapPoint> map_point_;  // 关联地图点

    bool is_outlier_ = false;       // 是否为异常点
    bool is_on_left_image_ = true;  // 标识是否提在左图，false为右图

   public:
    Feature() {}

    Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp)
        : frame_(frame), position_(kp) {}
};
}  // namespace my_VO

#endif  // MY_VO_FEATURE_H
