// undistort.cpp
// 2023 SEP 23
// Tershire


#include <iostream>

#include <opencv2/opencv.hpp>


int main(int argc, char **argv)
{
    // load image /////////////////////////////////////////////////////////////
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    // check loading
    if (image.data == nullptr)
    {
        std::cerr << "file" << argv[1] << " does not exist." << std::endl;
        return 0;
    }

    // image info.
    const int NUM_ROWS = image.rows;
    const int NUM_COLS = image.cols;

    // load camera info. //////////////////////////////////////////////////////
    std::string setting_file_path = "./data/calibration_setting.yaml";

    cv::FileStorage file = cv::FileStorage(setting_file_path, cv::FileStorage::READ);

    // check loading
    if (!file.isOpened())
    {
        std::cerr << "[ERROR]: could not open setting file at: " 
                  << setting_file_path << std::endl;
        return -1;
    }

    // cv::FileNode node_cameraMatrix = file["K"];
    // cv::FileNode node_distCoeffs   = file["D"];

    cv::Mat cameraMatrix = file["K"].mat();
    cv::Mat distCoeffs   = file["D"].mat();

    std::cout << "cameraMatrix: " << cameraMatrix << std::endl;
    std::cout << "distCoeffs  : " << distCoeffs   << std::endl;

    // extract
    double fx, fy, cx, cy;
    fx = cameraMatrix.at<double>(0, 0);
    fy = cameraMatrix.at<double>(1, 1);
    cx = cameraMatrix.at<double>(0, 2);
    cy = cameraMatrix.at<double>(1, 2); 

    double k1, k2, p1, p2;
    k1 = distCoeffs.at<double>(0);
    k2 = distCoeffs.at<double>(1);
    p1 = distCoeffs.at<double>(2);
    p2 = distCoeffs.at<double>(3);

    // undistort //////////////////////////////////////////////////////////////
    // <M1> custom code =======================================================
    cv::Mat image_undistorted = cv::Mat(NUM_ROWS, NUM_COLS, CV_8UC1);

    // pinhole unprojection and distorted projection
    double x, y;
    double r, r2, r4;
    double x_distorted, y_distorted, u_distorted, v_distorted;
    for (int v = 0; v < NUM_ROWS; ++v)
    {
        for (int u = 0; u < NUM_COLS; ++u)
        {
            x = (u - cx) / fx;
            y = (v - cy) / fy;

            r = sqrt(x * x + y * y);
            r2 = r  * r;
            r4 = r2 * r2;

            x_distorted = x * (1 + k1 * r2 + k2 * r4) + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
            y_distorted = y * (1 + k1 * r2 + k2 * r4) + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
            u_distorted = fx * x_distorted + cx;
            v_distorted = fy * y_distorted + cy;

            // assignment (nearest neighbor interpolation)
            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < NUM_COLS && v_distorted < NUM_ROWS)
            {
                image_undistorted.at<uchar>(v, u) = image.at<uchar>((int) v_distorted, (int) u_distorted);
            } 
            else 
            {
                image_undistorted.at<uchar>(v, u) = 0;
            }
        }
    }

    // <M2> OpenCV ============================================================
    cv::Mat image_undistorted_cv;
    cv::undistort(image, image_undistorted_cv, cameraMatrix, distCoeffs);

    // result /////////////////////////////////////////////////////////////////
    cv::imshow("image", image);
    cv::imshow("image_undistorted_custom", image_undistorted);
    cv::imshow("image_undistorted_OpenCV", image_undistorted_cv);
    cv::waitKey(0);

    return 0;
}
