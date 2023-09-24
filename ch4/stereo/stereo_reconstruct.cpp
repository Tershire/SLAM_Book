// undistort.cpp
// 2023 SEP 23
// Tershire

// run: ch4$ build/stereo/stereo_reconstruct


#include <iostream>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <pangolin/pangolin.h>

// prototype
void show_point_cloud(const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& pointcloud);


int main(int argc, char **argv)
{
    // load image /////////////////////////////////////////////////////////////
    cv::Mat image_L = cv::imread("./data/left.png" , cv::IMREAD_GRAYSCALE);
    cv::Mat image_R = cv::imread("./data/right.png", cv::IMREAD_GRAYSCALE);

    // camera calibration setting /////////////////////////////////////////////
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    double b = 0.573; // baseline

    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);
    
    cv::Mat disparity_sgbm, disparity;
    sgbm->compute(image_L, image_R, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0F);

    // point cloud ////////////////////////////////////////////////////////////
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> point_cloud;

    double x, y;
    double depth;
    for (int v = 0; v < image_L.rows; ++v)
        for (int u = 0; u < image_L.cols; ++u) 
        {
            if (disparity.at<float>(v, u) <= 0.0 || disparity.at<float>(v, u) >= 96.0) continue; // (?) what's the logic

            Eigen::Vector4d point(0, 0, 0, image_L.at<uchar>(v, u) / 255.0); // (x, y, z, color)

            // calculate position of point based on binocular model
            x = (u - cx) / fx;
            y = (v - cy) / fy;
            depth = fx * b / (disparity.at<float>(v, u));

            point[0] = x * depth;
            point[1] = y * depth;
            point[2] = depth;

            point_cloud.push_back(point);
        }

    cv::imshow("disparity", disparity / 96.0);
    cv::waitKey(0);
    
    // draw
    show_point_cloud(point_cloud);

    return 0;
}

void show_point_cloud(const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& point_cloud) 
{
    if (point_cloud.empty())
    {
        std::cerr << "[ERROR] point cloud is empty" << std::endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));

    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0F / 768.0F)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0F, 1.0F, 1.0F, 1.0F);

        glPointSize(2);
        glBegin(GL_POINTS);

        for (auto& point: point_cloud)
        {
            glColor3f(point[3], point[3], point[3]);
            glVertex3d(point[0], point[1], point[2]);
        }

        glEnd();
        pangolin::FinishFrame();
    }

    return;
}
