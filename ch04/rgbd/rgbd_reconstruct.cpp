// rgbd_reconstruct.cpp
// 2023 SEP 24
// Tershire

// run:


#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <pangolin/pangolin.h>

#include <boost/format.hpp>

typedef Sophus::SE3d SE3;
typedef std::vector<SE3, Eigen::aligned_allocator<SE3>> trajectory_type;
typedef Eigen::Matrix<double, 6, 1> Vec6;

typedef Eigen::Vector3d Vec3;
typedef Eigen::Quaterniond Quaternion;


void show_point_cloud(const std::vector<Vec6, Eigen::aligned_allocator<Vec6>>& point_cloud);


int main(int argc, char **argv)
{
    // load images and camera extrinsics //////////////////////////////////////
    std::vector<cv::Mat> color_images, depth_images;
    trajectory_type T_wc_poses;

    std::ifstream input_stream("./data/T_wc_pose_data.txt");
    if (!input_stream)
    {
        std::cerr << "ERROR: file not found" << std::endl;
        return -1;
    }

    const int NUM_IMAGES = 5;
    for (int i = 0; i < NUM_IMAGES; ++i)
    {
        boost::format format("./data/%s/%d.%s");
        color_images.push_back(cv::imread((format % "color" % (i + 1) % "png").str()));
        depth_images.push_back(cv::imread((format % "depth" % (i + 1) % "pgm").str(), -1)); // (?) -1 to read original image

        // (+) good skill: putting data into array using reference
        double data[7] = {0};
        for (auto& datum : data)
            input_stream >> datum;

        SE3 T_wc(Quaternion(data[6], data[3], data[4], data[5]),
            Vec3(data[0], data[1], data[2]));

        T_wc_poses.push_back(T_wc);
    }

    // camera intrinsic ///////////////////////////////////////////////////////
    double fx, fy, cx, cy;
    fx = 518.0;
    fy = 519.0;
    cx = 325.5;
    cy = 253.5;
    
    // collect points to form cloud ///////////////////////////////////////////
    double depth_scale = 1E3;

    std::vector<Vec6, Eigen::aligned_allocator<Vec6>> point_cloud;
    point_cloud.reserve(1E6); // <- (?) is it necessary

    cv::Mat color_image, depth_image;
    for (int i = 0; i < NUM_IMAGES; ++i)
    {
        color_image = color_images[i];
        depth_image = depth_images[i];
        
        SE3 T_wc = T_wc_poses[i];

        for (int v = 0; v < color_image.rows; ++v)
            for (int u = 0; u < color_image.cols; ++u)
            {
                unsigned int depth = depth_image.ptr<unsigned short>(v)[u]; // (!?) ()[]
                if (depth == 0) continue; // 0 means no measurement

                Vec3 p3D_camera;
                p3D_camera[2] = double(depth) / depth_scale;
                p3D_camera[0] = (u - cx) * p3D_camera[2] / fx;
                p3D_camera[1] = (v - cy) * p3D_camera[2] / fy;

                Vec3 p3D_world = T_wc * p3D_camera;

                // (x, y, z, color_B, color_G, color_R)
                Vec6 point;
                point.head<3>() = p3D_world; // (+) good skill
                point[5] = color_image.data[v * color_image.step + u * color_image.channels()];     // B
                point[4] = color_image.data[v * color_image.step + u * color_image.channels() + 1]; // G
                point[3] = color_image.data[v * color_image.step + u * color_image.channels() + 2]; // R
                point_cloud.push_back(point);
            }
    }

    std::cout << "total number of points in cloud: " << point_cloud.size() << " points." << std::endl;

    // draw
    show_point_cloud(point_cloud);

    return 0;
}

void show_point_cloud(const std::vector<Vec6, Eigen::aligned_allocator<Vec6>>& point_cloud)
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
            glColor3f(point[3] / 255.0, point[4] / 255.0, point[5] / 255.0);
            glVertex3d(point[0], point[1], point[2]);
        }

        glEnd();
        pangolin::FinishFrame();
    }

    return;
}
