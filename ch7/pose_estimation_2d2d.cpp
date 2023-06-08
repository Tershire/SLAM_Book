// pose_estimation_2d2d.cpp
// 2023 JUN 08
// Tershire

// bash execute command: /ch7$ build/pose_estimation_2d2d data/1.png data/2.png

// HEADER FILE ////////////////////////////////////////////////////////////////
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

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

Point2d pixel2cam(const Point2d& p, const Mat& K);

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
    vector<DMatch> matches;
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
void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1,
                          std::vector<KeyPoint> keypoints_2,
                          std::vector<DMatch> matches,
                          Mat& R, Mat& t)
{
    // camera matrix, TUM Freiburg2
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    // convert tmatching point to vector<Point2f>
    vector<Point2f> points_1;
    vector<Point2f> points_2;

    for (int i = 0; i < (int) matches.size(); i++)
    {
        points_1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points_2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    // calculate fundamental matrix ===========================================
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(points_1, points_2, FM_8POINT);

    cout << "fundamental_matrix:" << endl << fundamental_matrix << endl;

    // calculate essential matrix =============================================
    Point2d principal_point(325.1, 249.7); // camera optical center,
                                           // TUM dataset calibration value
    double focal_length = 521;             // camera focal length, 
                                           // TUM dataset calibration value

    Mat essential_matrix;
    essential_matrix = findEssentialMat(points_1, points_2, 
                                        focal_length, principal_point);

    cout << "essential_matrix: " << endl << essential_matrix << endl;

    // calculate the homography matrix ========================================
    // <B> in this Ex., scene is not flat, so
    // homography matrix is of little significance
    Mat homography_matrix;
    homography_matrix = findHomography(points_1, points_2, RANSAC, 3);

    cout << "homography_matrix: " << endl << homography_matrix << endl;

    // recover rotation & translation from the essential matrix ===============
    recoverPose(essential_matrix, points_1, points_2, R, t, 
                focal_length, principal_point);
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;
}


// ============================================================================
Point2d pixel2cam(const Point2d &p, const Mat &K) 
{
    return Point2d((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                   (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}
