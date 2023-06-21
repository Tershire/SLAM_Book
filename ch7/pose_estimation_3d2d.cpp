// pose_estimation_3d2d.cpp
// 2023 JUN 08
// Tershire

// bash execute command:

// HEADER FILE ////////////////////////////////////////////////////////////////
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <eigen3/Eigen/Core>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <sophus/se3.hpp>

#include <chrono>

// using namespace std;
using namespace cv;


// TYPEDEF ////////////////////////////////////////////////////////////////////
// BA by g2o
typedef std::vector<Eigen::Vector2d, 
                    Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef std::vector<Eigen::Vector3d, 
                    Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;


// PROTOTYPE //////////////////////////////////////////////////////////////////
void find_feature_matches(const Mat& img_1, const Mat& img_2,
                          std::vector<KeyPoint>& keypoints_1,
                          std::vector<KeyPoint>& keypoints_2,
                          std::vector<DMatch>& matches_);

// pixel coordinates -> camera normalized coordinates
Point2d pixel2cam(const Point2d& p, const Mat& K);


// Bundle Adjustment ==========================================================
// BA by gauss-newton
void bundle_adjustment_Gauss_Newton(const VecVector3d& points_3d,
                                    const VecVector2d& points_2d,
                                    const Mat& K,
                                    Sophus::SE3d& pose);

void bundle_adjustment_g2o(const VecVector3d& points_3d,
                           const VecVector2d& points_2d,
                           const Mat& K,
                           Sophus::SE3d& pose);


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
    Mat K = (Mat_<double>(3, 3) << 520.9,   0  , 325.1, 
                                       0, 521.0, 249.7, 
                                       0,   0  ,   1  );

    std::vector<Point3f> points_3d;
    std::vector<Point2f> points_2d;

    for (DMatch m:matches) 
    {
        ushort depth = depth_map_1.ptr<unsigned short>
                                      (int(keypoints_1[m.queryIdx].pt.y))
                                      [int(keypoints_1[m.queryIdx].pt.x)];

        if (depth == 0)   // bad depth
            continue;

        float dd = depth / 5000.0; // man! what is dd
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        points_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
        points_2d.push_back(keypoints_2[m.trainIdx].pt);
    }

    //
    std::cout << "3d-2d pairs: " << points_3d.size() << std::endl;


    // cv::solvePnP() =========================================================
    // timer: ini -------------------------------------------------------------
    std::chrono::steady_clock::time_point t_ini = 
    std::chrono::steady_clock::now();

    // solve PnP --------------------------------------------------------------
    Mat r, t;
    solvePnP(points_3d, points_2d, K, Mat(), r, t, false);

    // convert to rotation matrix
    Mat R;
    cv::Rodrigues(r, R);

    // timer: fin -------------------------------------------------------------
    std::chrono::steady_clock::time_point t_fin = 
    std::chrono::steady_clock::now();
    std::chrono::duration<double> time_taken = 
    std::chrono::duration_cast<std::chrono::duration<double>>(t_fin - t_ini);
    std::cout << "Time cv::solvePnP: " << time_taken.count() << " [s].\n";

    //
    std::cout << "R = \n" << R << std::endl;
    std::cout << "t = \n" << t << std::endl;


    // Bundle Adjustment ======================================================
    // ??? --------------------------------------------------------------------
    VecVector3d points_3d_eigen;
    VecVector2d points_2d_eigen;
    for (size_t i = 0; i < points_3d.size(); i++) 
    {
        points_3d_eigen.push_back(Eigen::Vector3d(points_3d[i].x, 
                                                  points_3d[i].y, 
                                                  points_3d[i].z));
        points_2d_eigen.push_back(Eigen::Vector2d(points_2d[i].x, 
                                                  points_2d[i].y));
    }

    //
    std::cout << "calling bundle adjustment by Gauss-Newton" << std::endl;

    // [1] BA: Gauss-Newton ###################################################
    Sophus::SE3d pose_GN;
    
    // timer: ini -------------------------------------------------------------
    t_ini = std::chrono::steady_clock::now();

    // BA
    bundle_adjustment_Gauss_Newton(points_3d_eigen, points_2d_eigen, K, 
                                   pose_GN);
    
    // timer: fin -------------------------------------------------------------
    t_fin = std::chrono::steady_clock::now();
    time_taken = 
    std::chrono::duration_cast<std::chrono::duration<double>>(t_fin - t_ini);
    std::cout << "Time solve PnP by BA Gauss-Newton: " << time_taken.count() 
              << " [s].\n";

    // [2] BA: g2o ############################################################
    Sophus::SE3d pose_g2o;

    // timer: ini -------------------------------------------------------------
    t_ini = std::chrono::steady_clock::now();

    // BA
    bundle_adjustment_g2o(points_3d_eigen, points_2d_eigen, K, 
                          pose_g2o);
    
    // timer: fin -------------------------------------------------------------
    t_fin = std::chrono::steady_clock::now();
    time_taken = 
    std::chrono::duration_cast<std::chrono::duration<double>>(t_fin - t_ini);
    std::cout << "Time solve PnP by BA g2o: " << time_taken.count() 
              << " [s].\n";

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
        virtual void oplusImpl(const double *update) override 
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
class Edge_Projection: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, 
                                                 Vertex_Pose> 
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Edge_Projection(const Eigen::Vector3d& pos, const Eigen::Matrix3d& K): 
                        _pos3d(pos), _K(K) {}

        virtual void computeError() override 
        {
            const Vertex_Pose* v = static_cast<Vertex_Pose*> (_vertices[0]);
            Sophus::SE3d T = v->estimate();
            Eigen::Vector3d pos_pixel = _K * (T * _pos3d);
            pos_pixel /= pos_pixel[2];
            _error = _measurement - pos_pixel.head<2>();
        }

        virtual void linearizeOplus() override 
        {
            const Vertex_Pose* v = static_cast<Vertex_Pose*> (_vertices[0]);
            Sophus::SE3d T = v->estimate();
            Eigen::Vector3d pos_cam = T * _pos3d;
            double fx = _K(0, 0);
            double fy = _K(1, 1);
            double cx = _K(0, 2);
            double cy = _K(1, 2);
            double X = pos_cam[0];
            double Y = pos_cam[1];
            double Z = pos_cam[2];
            double Z2 = Z * Z;

            _jacobianOplusXi // ?
            << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, 
               -fx - fx * X * X / Z2, fx * Y / Z,
                 0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, 
               -fy * X * Y / Z2, -fy * X / Z;
        }

        virtual bool  read(std::istream& in )       override {}
        virtual bool write(std::ostream& out) const override {}

    private:
        Eigen::Vector3d _pos3d;
        Eigen::Matrix3d _K;
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


// XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
Point2d pixel2cam(const Point2d &p, const Mat &K) 
{
    return Point2d((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                   (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}


// XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
void bundle_adjustment_Gauss_Newton(const VecVector3d& points_3d,
                                    const VecVector2d& points_2d,
                                    const Mat& K,
                                    Sophus::SE3d& pose)
{
    // Setting ================================================================
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    
    double cost = 0, last_cost = 0;
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    const int NUM_ITERATION = 10;


    // ??? ====================================================================
    for (int k = 0; k < NUM_ITERATION; k++) 
    {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero(); // what is b ?

        // compute cost
        cost = 0;
        for (int i = 0; i < points_3d.size(); i++) 
        {
            Eigen::Vector3d pc = pose * points_3d[i]; // what is pc ?
            double inv_z  = 1.0 / pc[2];
            double inv_z2 = inv_z * inv_z;
            Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, 
                                 fy * pc[1] / pc[2] + cy);

            Eigen::Vector2d e = points_2d[i] - proj;

            cost += e.squaredNorm();
            Eigen::Matrix<double, 2, 6> J;
            J << -fx * inv_z,
                0,
                 fx * pc[0] * inv_z2,
                 fx * pc[0] * pc[1] * inv_z2,
                -fx - fx * pc[0] * pc[0] * inv_z2,
                 fx * pc[1] * inv_z,
                0,
                -fy * inv_z,
                 fy * pc[1] * inv_z2,
                 fy + fy * pc[1] * pc[1] * inv_z2,
                -fy * pc[0] * pc[1] * inv_z2,
                -fy * pc[0] * inv_z;

            H +=  J.transpose() * J;
            b += -J.transpose() * e;
        }

        Vector6d dx;
        dx = H.ldlt().solve(b);

        if (std::isnan(dx[0])) 
        {
            std::cout << "result is NaN!" << std::endl;
            break;
        }

        if (k > 0 && cost >= last_cost) 
        {
            // cost increase, update is not good
            std::cout << "cost: " << cost << ", last cost: " << last_cost
                      << std::endl;
        break;
        }

        // update your estimation
        pose = Sophus::SE3d::exp(dx) * pose;
        last_cost = cost;

        std::cout << "NUM_ITERATION: " << k << " cost = " 
                  << std::setprecision(12) << cost << std::endl;
        if (dx.norm() < 1e-6) 
        {
            // converge
            break;
        }
    }

    std::cout << "pose by g-n: \n" << pose.matrix() << std::endl;
}


// XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
void bundle_adjustment_g2o(const VecVector3d& points_3d,
                           const VecVector2d& points_2d,
                           const Mat& K,
                           Sophus::SE3d& pose)
{
    // Setting ================================================================
    // pose: 6, landmark: 3 (?)
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> 
            BlockSolverType;  
    
    // linear solver type
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> 
            LinearSolverType;

    // gradient descent method
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
                      std::make_unique<BlockSolverType>
                     (std::make_unique<LinearSolverType>()));
                    //   g2o::make_unique<BlockSolverType>
                    //  (g2o::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer optimizer; // graph model
    optimizer.setAlgorithm(solver); // set up the solver
    optimizer.setVerbose(true);     // turn on debug output

    // vertex
    Vertex_Pose* vertex_pose = new Vertex_Pose(); // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(vertex_pose);

    // K
    Eigen::Matrix3d K_eigen;
    K_eigen << K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
               K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
               K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

    // edges
    int index = 1;
    for (size_t i = 0; i < points_2d.size(); i++)
    {
        auto p2d = points_2d[i];
        auto p3d = points_3d[i];

        Edge_Projection* edge = new Edge_Projection(p3d, K_eigen);

        edge->setId(index);
        edge->setVertex(0, vertex_pose);
        edge->setMeasurement(p2d);
        edge->setInformation(Eigen::Matrix2d::Identity());

        optimizer.addEdge(edge);

        index++;
    }

    // timer: ini -------------------------------------------------------------
    std::chrono::steady_clock::time_point t_ini = 
    std::chrono::steady_clock::now();

    // ??? --------------------------------------------------------------------
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    // timer: fin -------------------------------------------------------------
    std::chrono::steady_clock::time_point t_fin = 
    std::chrono::steady_clock::now();
    std::chrono::duration<double> time_taken = 
    std::chrono::duration_cast<std::chrono::duration<double>>(t_fin - t_ini);
    std::cout << "Time g2o optimization: " << time_taken.count() 
              << " [s].\n";

    //
    std::cout << "pose estimated by g2o =\n" 
              << vertex_pose->estimate().matrix() << std::endl;
    
    //
    pose = vertex_pose->estimate();
}

