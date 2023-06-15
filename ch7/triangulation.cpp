// triangulation.cpp
// 2023 JUN 15
// Tershire

// bash execute command: 

// HEADER FILE ////////////////////////////////////////////////////////////////
#include <iostream>
#include <opencv2/opencv.hpp>

// using namespace std;
using namespace cv;


// PROTOTYPE //////////////////////////////////////////////////////////////////
void find_feature_matches(const Mat& img_1, const Mat& img_2,
                          std::vector<KeyPoint>& keypoints_1,
                          std::vector<KeyPoint>& keypoints_2,
                          std::vector<DMatch>& matches_);

void pose_estimation_2d2d(std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches,
                          Mat& R, Mat& t);

void triangulation(const std::vector<KeyPoint>& keypoint_1,
                   const std::vector<KeyPoint>& keypoint_2,
                   const std::vector<DMatch>& matches,
                   const Mat& R, const Mat& t,
                   std::vector<Point3d>& points);

// pixel coordinates -> camera normalized coordinates
Point2d pixel2cam(const Point2d& p, const Mat& K);

//
inline cv::Scalar get_color(float depth)
{
    float lower_threshold = 10, upper_threshold = 50; 
    float threshold_range = upper_threshold - lower_threshold;

    if (depth > upper_threshold) depth = upper_threshold;
    if (depth < lower_threshold) depth = lower_threshold;

    return cv::Scalar(255 * depth / threshold_range,
                      0,
                      255 * (1 - depth / threshold_range));
}


// MAIN ///////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // ?
    if (argc != 3)
    {
        std::cout << "usage: feature extraction img1 img2\n";
        return 1;
    }


    // Setting ================================================================
    // read images
    Mat img_1 = imread(argv[1], IMREAD_COLOR);
    Mat img_2 = imread(argv[2], IMREAD_COLOR);
    assert(img_1.data != nullptr && img_2.data != nullptr);


    // Find Feature Matches ===================================================
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    std::vector<DMatch> matches;

    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);

    std::cout << "Number of matches: " << matches.size() << std::endl;


    // Estimate Motion Between Two Images =====================================
    Mat R, t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    
    // Triangulation ==========================================================
    std::vector<Point3d> points;
    triangulation(keypoints_1, keypoints_2, matches, R, t, points);

    
    // Validation =============================================================
    // verify: reprojection relationship between triangulated point and 
    //                                           feature      point
    // what is Mat_ ?
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    Mat img_1_plot = img_1.clone();
    Mat img_2_plot = img_2.clone();
    for (int i = 0; i < matches.size(); i++)
    {
        // .pt (?)
        // 1st image
        float depth_1 = points[i].z;
        std::cout << "depth 1: " << depth_1 << std::endl;
        Point2d pt1 = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
        cv::circle(img_1_plot, keypoints_1[matches[i].queryIdx].pt, 2, 
                   get_color(depth_1), 2);

        // 2nd image (?)
        Mat pt2 = R * (Mat_<double>(3, 1) << points[i].x, 
                                             points[i].y, 
                                             points[i].z) + t;
        float depth_2 = pt2.at<double>(2, 0);
        cv::circle(img_2_plot, keypoints_2[matches[i].trainIdx].pt, 2, 
                   get_color(depth_2), 2);
    }

    cv::imshow("img 1", img_1_plot);
    cv::imshow("img 2", img_2_plot);
    cv::waitKey();    

    return 0;
}


// HELPER /////////////////////////////////////////////////////////////////////
// ============================================================================
void find_feature_matches(const Mat& img_1, const Mat& img_2,
                          std::vector<KeyPoint>& keypoints_1,
                          std::vector<KeyPoint>& keypoints_2,
                          std::vector<DMatch>& matches_)
{
    // setup
    Mat descriptors_1, descriptors_2;

    Ptr<FeatureDetector>     detector   = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = 
        DescriptorMatcher::create("BruteForce-Hamming");


    // ORB Extraction =========================================================    
    // Keypoint  : detect Oriented FAST
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    // Descriptor: compute BRIEF Descriptor
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);


    // Feature Matching =======================================================
    std::vector<DMatch> matches;
    matcher->match(descriptors_1, descriptors_2, matches);

    // match point pair filtering
    double min_dist = 10000, max_dist = 0;

    // find max and min distances
    double dist;
    for (int i = 0; i < descriptors_1.rows; i++) {
        dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    // criteria
    for (int i = 0; i < descriptors_1.rows; i++) 
    {
        if (matches[i].distance <= max(2 * min_dist, 30.0))
        {
            matches_.push_back(matches[i]);
        }
    }
}


// ============================================================================
void pose_estimation_2d2d(std::vector<KeyPoint>& keypoints_1,
                          std::vector<KeyPoint>& keypoints_2,
                          std::vector<DMatch>& matches,
                          Mat& R, Mat& t)
{
    // camera matrix, TUM Freiburg2
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    // convert tmatching point to vector<Point2f>
    std::vector<Point2f> points_1;
    std::vector<Point2f> points_2;

    for (int i = 0; i < (int) matches.size(); i++)
    {
        points_1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points_2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    // calculate essential matrix =============================================
    Point2d principal_point(325.1, 249.7); // camera optical center,
                                           // TUM dataset calibration value
    double focal_length = 521;             // camera focal length, 
                                           // TUM dataset calibration value

    Mat essential_matrix;
    essential_matrix = findEssentialMat(points_1, points_2, 
                                        focal_length, principal_point);


    // recover rotation & translation from the essential matrix ===============
    recoverPose(essential_matrix, points_1, points_2, R, t, 
                focal_length, principal_point);
}


// ============================================================================
void triangulation(const std::vector<KeyPoint>& keypoint_1,
                   const std::vector<KeyPoint>& keypoint_2,
                   const std::vector<DMatch>& matches,
                   const Mat& R, const Mat& t,
                   std::vector<Point3d>& points)
{
    // (?)
    Mat T1 = (Mat_<float>(3, 4) <<
              1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0);

    Mat T2 = (Mat_<float>(3, 4) <<
              R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), 
              t.at<double>(0, 0),
              R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), 
              t.at<double>(1, 0),
              R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), 
              t.at<double>(2, 0));

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    std::vector<Point2f> points_1, points_2;
    for (DMatch m:matches) 
    {
        // pixel coordinates -> camera normalized coordinates
        points_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
        points_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
    }

    Mat points_4d;
    cv::triangulatePoints(T1, T2, points_1, points_2, points_4d);

    // homogenous coordinates -> inhomogenous coordinates
    for (int i = 0; i < points_4d.cols; i++) 
    {
        Mat x = points_4d.col(i);
        x /= x.at<float>(3, 0); // normalize
        Point3d p(x.at<float>(0, 0),
                  x.at<float>(1, 0),
                  x.at<float>(2, 0));
        points.push_back(p);
    }
}


// ============================================================================
Point2d pixel2cam(const Point2d &p, const Mat &K) 
{
    return Point2d((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                   (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}
