// pose_estimation_3d3d.cpp
// 2023 JUN 27
// Tershire

// fix: g2o.make_unique() -> std.make_unique() (is this okay ?)

// bash execute command: /ch7$ build/pose_estimation_3d3d data/1.png data/2.png
//                             data/1_depth.png data/2_depth.png


// HEADER FILE ////////////////////////////////////////////////////////////////
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/SVD>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <sophus/se3.hpp>

#include <chrono>

// using namespace std;
using namespace cv;


// PROTOTYPE //////////////////////////////////////////////////////////////////
void find_feature_matches(const Mat& img_1, const Mat& img_2,
                          std::vector<KeyPoint>& keypoints_1,
                          std::vector<KeyPoint>& keypoints_2,
                          std::vector<DMatch>& matches_);

// pixel coordinates -> camera normalized coordinates
Point2d pixel2cam(const Point2d& p, const Mat& K);

void pose_estimation_3d3d(const std::vector<Point3f>& points_1,
                          const std::vector<Point3f>& points_2,
                          Mat& R, Mat& t);
                          

// Bundle Adjustment ==========================================================
// BA
void bundle_adjustment(const std::vector<Point3f>& points_1,
                       const std::vector<Point3f>& points_2,
                       Mat& R, Mat& t);


// MAIN ///////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // user input check
    if (argc != 5)
    {
        std::cout << "usage: pose_estimation_3d2d img1 img2 depth1 depth2\n";
        return 1;
    }


    // Load Image =============================================================
    Mat img_1 = imread(argv[1], IMREAD_COLOR);
    Mat img_2 = imread(argv[2], IMREAD_COLOR);
    assert(img_1.data && img_2.data && "Can not load images!");


    // Find Feature Matches ===================================================
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    std::vector<DMatch> matches;

    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);

    std::cout << "Number of matches: " << matches.size() << std::endl;


    // Create 3D Points =======================================================
    Mat depth_map_1 = imread(argv[3], IMREAD_UNCHANGED);
    Mat depth_map_2 = imread(argv[4], IMREAD_UNCHANGED);
    Mat K = (Mat_<double>(3, 3) << 520.9,   0  , 325.1, 
                                       0, 521.0, 249.7, 
                                       0,   0  ,   1  );

    std::vector<Point3f> points_3d_1, points_3d_2;

    for (DMatch m:matches) 
    {
        ushort depth_1 = depth_map_1.ptr<unsigned short>
                                        (int(keypoints_1[m.queryIdx].pt.y))
                                        [int(keypoints_1[m.queryIdx].pt.x)];
        ushort depth_2 = depth_map_2.ptr<unsigned short>
                                        (int(keypoints_2[m.trainIdx].pt.y))
                                        [int(keypoints_2[m.trainIdx].pt.x)];

        if (depth_1 == 0 || depth_2 == 0) // bad depth
            continue;

        float dd1 = float(depth_1) / 5000.0; // man! what is dd
        float dd2 = float(depth_2) / 5000.0; 
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K); // why queryIdx
        Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K); // why trainIdx
        points_3d_1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
        points_3d_2.push_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));
    }

    //
    std::cout << "3d-3d pairs: " << points_3d_1.size() << std::endl;

    //
    Mat R, t;
    pose_estimation_3d3d(points_3d_1, points_3d_2, R, t);
    std::cout << "ICP via SVD results:\n"
              << "R     = " <<  R         << std::endl
              << "t     = " <<  t         << std::endl
              << "R_inv = " <<  R.t()     << std::endl
              << "t_inv = " << -R.t() * t << std::endl;
    std::cout << std::endl;
    
    std::cout << "Calling bundle adjustment\n";
    bundle_adjustment(points_3d_1, points_3d_2, R, t);

    // verify P1 = R * P2 + t // P2 = R * P1 + t, no ?
    for (int i = 0; i < 5; i++) 
    {
        std::cout << "P1 = " << points_3d_1[i] << std::endl
                  << "P2 = " << points_3d_2[i] << std::endl;
        std::cout << "(R*P2 + t) = "
                  << R * (Mat_<double>(3, 1) << points_3d_2[i].x, 
                                                points_3d_2[i].y, 
                                                points_3d_2[i].z) + t
                  << std::endl;
    }

    return 0;
}


// CLASS //////////////////////////////////////////////////////////////////////
// classes for {Vertex, Edge} used in g2o BA

// XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
class Vertex_Pose: public g2o::BaseVertex<6, Sophus::SE3d> 
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW; // ?

        virtual void setToOriginImpl() override 
        {
            _estimate = Sophus::SE3d();
        }

        // left multiplication on SE3
        virtual void oplusImpl(const double* update) override 
        {
            Eigen::Matrix<double, 6, 1> update_eigen;
            update_eigen << update[0], update[1], update[2], 
                            update[3], update[4], update[5];
            _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
        }

        virtual bool  read(std::istream& in )       override {}
        virtual bool write(std::ostream& out) const override {}
};

// XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
class Edge_Projection_XYZ_RGBD_Pose_Only: public g2o::BaseUnaryEdge<3, 
                                                 Eigen::Vector3d, 
                                                 Vertex_Pose> 
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Edge_Projection_XYZ_RGBD_Pose_Only(const Eigen::Vector3d& point):
                                           _point(point) {}

        virtual void computeError() override
        {
            const Vertex_Pose* vertex_pose = static_cast<const Vertex_Pose*> 
                                      (_vertices[0]);
            _error = _measurement - vertex_pose->estimate() * _point;
        }

        virtual void linearizeOplus() override
        {
            Vertex_Pose* vertex_pose = static_cast<Vertex_Pose*>(_vertices[0]);
            Sophus::SE3d T = vertex_pose->estimate();
            Eigen::Vector3d xyz_trans = T * _point;
            _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
            _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(xyz_trans);
        }

        bool  read(std::istream &in )       {}
        bool write(std::ostream &out) const {}

    protected:
        Eigen::Vector3d _point;
};


// HELPER /////////////////////////////////////////////////////////////////////
// XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
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
    double min_dist = 10000, max_dist = 0; // swapped?

    // find max and min distances
    double dist;
    for (int i = 0; i < descriptors_1.rows; i++)
    {
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

// XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
void pose_estimation_3d3d(const std::vector<Point3f>& points_1,
                          const std::vector<Point3f>& points_2,
                          Mat& R, Mat& t)
{
    // p
    Point3f p1, p2;
    const int N = points_1.size();
    for (int i = 0; i < N; i++) // ?
    {
        p1 += points_1[i];
        p2 += points_2[i];
    }
    p1 = Point3f(Vec3f(p1) / N); // ?
    p2 = Point3f(Vec3f(p2) / N);

    // q
    std::vector<Point3f> q1(N), q2(N);
    for (int i = 0; i < N; i++)
    {
        q1[i] = points_1[i] - p1;
        q2[i] = points_2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i++)
    {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) 
           * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    std::cout << "W = " << W << std::endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU 
                                           | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    std::cout << "U = " << U << std::endl;
    std::cout << "V = " << V << std::endl;

    Eigen::Matrix3d R_ = U * (V.transpose()); // why R_ , and not R ?
    if (R_.determinant() < 0)
    {
        R_ = -R_;
    }
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - 
                    R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    // convert to cv::Mat
    R = (Mat_<double>(3, 3) <<
        R_(0, 0), R_(0, 1), R_(0, 2),
        R_(1, 0), R_(1, 1), R_(1, 2),
        R_(2, 0), R_(2, 1), R_(2, 2));

    t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

// XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
Point2d pixel2cam(const Point2d &p, const Mat &K) 
{
    return Point2d((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                   (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

// XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
void bundle_adjustment(const std::vector<Point3f>& points_1,
                       const std::vector<Point3f>& points_2,
                       Mat& R, Mat& t)
{
    // Setting ================================================================
    // build graph optimization, first set g2o
    typedef g2o::BlockSolverX BlockSolverType;

    // linear solver type
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> 
            LinearSolverType;

    // gradient descent method
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
    std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer; // graph model
    optimizer.setAlgorithm(solver); // set up the solver
    optimizer.setVerbose(true);     // turn on debug output

    // vertex
    Vertex_Pose* vertex_pose = new Vertex_Pose(); // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(vertex_pose);

    // edges
    for (size_t i = 0; i < points_1.size(); i++)
    {
        Edge_Projection_XYZ_RGBD_Pose_Only* edge = 
        new Edge_Projection_XYZ_RGBD_Pose_Only(Eigen::Vector3d(points_2[i].x, 
                                                               points_2[i].y, 
                                                               points_2[i].z));
        // ^ why points_2 , and not points_1 ?

        edge->setVertex(0, vertex_pose);
        edge->setMeasurement(Eigen::Vector3d(points_1[i].x, 
                                             points_1[i].y, 
                                             points_1[i].z));
        edge->setInformation(Eigen::Matrix3d::Identity());
        optimizer.addEdge(edge);
    }

    // timer: ini -------------------------------------------------------------
    std::chrono::steady_clock::time_point t_ini = 
    std::chrono::steady_clock::now();

    // optimization ===========================================================
    // optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    // ========================================================================

    // timer: fin -------------------------------------------------------------
    std::chrono::steady_clock::time_point t_fin = 
    std::chrono::steady_clock::now();
    std::chrono::duration<double> time_taken = 
    std::chrono::duration_cast<std::chrono::duration<double>>(t_fin - t_ini);
    std::cout << "Time optimization: " << time_taken.count() 
              << " [s].\n";

    //
    std::cout << "pose estimated by g2o =\n" 
              << vertex_pose->estimate().matrix() << std::endl;
    
    // convert to cv::Mat
    Eigen::Matrix3d R_ = vertex_pose->estimate().rotationMatrix();
    Eigen::Vector3d t_ = vertex_pose->estimate().translation();
    R = (Mat_<double>(3, 3) <<
        R_(0, 0), R_(0, 1), R_(0, 2),
        R_(1, 0), R_(1, 1), R_(1, 2),
        R_(2, 0), R_(2, 1), R_(2, 2));

    t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}
