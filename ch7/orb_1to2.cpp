// orb_1to2.cpp
// 2023 MAY 20
// Tershire

// bash execute command: ch7$ build/orb_1to2 Data/1.png Data/2.png

// HEADER FILE ////////////////////////////////////////////////////////////////
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

// MAIN ///////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    using namespace std;
    using namespace cv;

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

    // setup
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = 
        DescriptorMatcher::create("BruteForce-Hamming");


    // ORB Extraction =========================================================
    // timer: ini -------------------------------------------------------------
    chrono::steady_clock::time_point t_ini = chrono::steady_clock::now();
    
    // Keypoint  : detect Oriented FAST
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    // Descriptor: compute BRIEF Descriptor
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    // timer: fin -------------------------------------------------------------
    chrono::steady_clock::time_point t_fin = chrono::steady_clock::now();
    chrono::duration<double> time_taken = 
        chrono::duration_cast<chrono::duration<double>>(t_fin - t_ini);
    cout << "Time ORB Extraction: " << time_taken.count() << " [s].\n";

    // output -----------------------------------------------------------------
    Mat outImg_1;
    drawKeypoints(img_1, keypoints_1, outImg_1, Scalar::all(-1),
                  DrawMatchesFlags::DEFAULT);
    imshow("ORB Features", outImg_1);


    // Feature Matching =======================================================
    // timer: ini -------------------------------------------------------------
    t_ini = chrono::steady_clock::now();

    // Hamming Distance -------------------------------------------------------
    vector<DMatch> matches;
    matcher->match(descriptors_1, descriptors_2, matches);

    // timer: fin -------------------------------------------------------------
    t_fin = chrono::steady_clock::now();
    time_taken = 
        chrono::duration_cast<chrono::duration<double>>(t_fin - t_ini);
    cout << "Time ORB Feature Matching: " << time_taken.count() << " [s].\n";

    // sort & remove outliers -------------------------------------------------
    // distance [min, max]
    auto min_max = minmax_element(matches.begin(), matches.end(),
        [](const DMatch &m1, const DMatch &m2) 
        {return m1.distance < m2.distance;}); // hard to understand
    double min_distance = min_max.first ->distance;
    double max_distance = min_max.second->distance;

    printf("Max Distance: %f \n", max_distance);
    printf("Min Distance: %f \n", min_distance);

    // remove bad matching
    // in this case, when Hamming Distance is less than twice of min
    std::vector<DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        if (matches[i].distance <= max(2 * min_distance, 30.0)) // 30.0?
        {
            good_matches.push_back(matches[i]);
        }
    }


    // Draw Output ============================================================
    Mat img_match;
    Mat img_good_match; 
    
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, 
                matches, img_match);
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, 
                img_good_match);
    
    imshow("all matches" , img_match);
    imshow("good matches", img_good_match);
    waitKey(0);

    return 0;
}
