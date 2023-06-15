// triangulation.cpp
// 2023 JUN 15
// Tershire

// bash execute command: 

// HEADER FILE ////////////////////////////////////////////////////////////////
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


// PROTOTYPE //////////////////////////////////////////////////////////////////
void find_feature_matches(const Mat& img_1, const Mat& img_2,
                          std::vector<KeyPoint>& keypoints_1,
                          std::vector<KeyPoint>& keypoints_2,
                          std::vector<DMatch>& matches_);

void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1,
                          std::vector<KeyPoint> keypoints_2,
                          std::vector<DMatch> matches,
                          Mat& R, Mat& t);

// pixel coordinates -> camera normalized coordinates
Point2d pixel2cam(const Point2d& p, const Mat& K);

void triangulation(const vector<KeyPoint> &keypoint_1,
                    const vector<KeyPoint> &keypoint_2,
                    const std::vector<DMatch> &matches,
                    const Mat &R, const Mat &t,
                    vector<Point3d> &points);

//
inline cv::Scalar get_color(float depth)
{
    float lower_threshold = 10, upper_threshold = 50; 
    float threshold_range = upper_threshold - lower_threshold;

    if (depth > upper_threshold) depth = upper_threshold;
    if (depth < lower_threshold) depth = lower_threshold;

    return cv::Scalar(255 * depth / threshold_range,
                      0
                      255 * (1 - depth / threshold_range));
}

// MAIN ///////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // ?
    if (argc != 3)
    {
        cout << "usage: feature extraction img1 img2\n";
        return 1;
    }


    // Setting ================================================================
    // read images
    Mat img_1 = imread(argv[1], IMREAD_COLOR);
    Mat img_2 = imread(argv[2], IMREAD_COLOR);
    assert(img_1.data != nullptr && img_2.data != nullptr);


    // Find Feature Matches ===================================================
    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;

    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);

    cout << "Number of matches: " << matches.size() << endl;


    // Estimate Motion Between Two Images =====================================
    Mat R, t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    
    // Validation =============================================================
    // verrify: E = t ^ R * scale (?)
    Mat t_x =
    (Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
     t.at<double>(2, 0), 0, -t.at<double>(0, 0),
    -t.at<double>(1, 0), t.at<double>(0, 0), 0);

    cout << "t ^ R = " << endl << t_x * R << endl;

    // validate: epipolar constraint
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    for (DMatch m: matches) 
    {
        Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);

        Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);

        Mat d = y2.t() * t_x * R * y1;

        cout << "Epipolar constraint = " << d << endl;
    }

    return 0;
}


// HELPER /////////////////////////////////////////////////////////////////////
// ============================================================================