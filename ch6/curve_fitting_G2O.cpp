// curve_fitting_G2O.cpp
// 2023 JUL 20
// Tershire


#include <iostream>

#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <Eigen/Core>

#include <opencv2/core/core.hpp>

#include <cmath>
#include <chrono>


// Vertex: 3D =================================================================
class Curve_Fitting_Vertex: public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // override ---------------------------------------------------------------
    // reset
    virtual void setToOriginImpl() override
    {
        _estimate << 0, 0, 0;
    }

    // plus operator
    virtual void oplusImpl(const double* update) override
    {
        _estimate += Eigen::Vector3d(update);
    }

    // ------------------------------------------------------------------------
    virtual bool read(std::istream& in) {}

    virtual bool write(std::ostream& out) const {}
};


// Edge: 1D ===================================================================
class Curve_Fitting_Edge: public g2o::BaseUnaryEdge<1, double, Curve_Fitting_Vertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // constructor & destructor -----------------------------------------------
    Curve_Fitting_Edge(double x): BaseUnaryEdge(), _x(x) {}

    // override ---------------------------------------------------------------
    // define error calculation
    virtual void computeError() override
    {
        // what is this (?) why type cast (?)
        const Curve_Fitting_Vertex* vertex = 
            static_cast<const Curve_Fitting_Vertex*>(_vertices[0]);

        // update optimization variable estimation
        const Eigen::Vector3d abc = vertex->estimate();

        // why abc(i, j) format (?) Eigen (?)
        _error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
    }

    // Jacobian
    virtual void linearizeOplus() override
    {
        const Curve_Fitting_Vertex* vertex =
            static_cast<const Curve_Fitting_Vertex*>(_vertices[0]);

        const Eigen::Vector3d abc = vertex->estimate();

        // why abc[i] format (?)
        double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);

        // is it manually calculated (?)
        _jacobianOplusXi[0] = -_x * _x * y;
        _jacobianOplusXi[1] = -_x * y;
        _jacobianOplusXi[2] = -y;
    }

    // ------------------------------------------------------------------------
    virtual bool read(std::istream& in) {}

    virtual bool write(std::ostream& out) const {}

    // field ------------------------------------------------------------------
    double _x; // x data, note y is given as _measurement
};


int main(int argc, char **argv)
{
    // SETTING ================================================================
    double a_true = 1.0, b_true =  2.0, c_true = 1.0; // true     value
    double a_est  = 2.0, b_est  = -1.0, c_est  = 5.0; // estimate value

    int N = 100; // number of data points

    double w_sigma = 1.0; // Gaussian distribution standard deviation (sigma)
    double inv_w_sigma = 1.0 / w_sigma;
    cv::RNG rng; // OpenCV random number generator


    // PREPARE DATA POINTS (x, y)s ============================================
    std::vector<double> x_data, y_data;
    for (int i = 0; i < N; i++)
    {
        double x = i / 100.0; // <!> CAREFUL: very wrong result if just put 100, not 100.0 <!>
        x_data.push_back(x);
        y_data.push_back(exp(a_true*x*x + b_true*x + c_true) 
                         + rng.gaussian(w_sigma * w_sigma));
    }


    // SOLVER SETTING =========================================================
    // Typedef for Solver -----------------------------------------------------
    // <optimization variable: 3D, error: 1D>
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;
    // linear solver type
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    // choose solver algorithm: {gradient descent, <G-M>, <L-M>, DogLeg} ------
    // what std::make_unique does (?)
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));


    // GRAPH OPTIMIZER SETTING ================================================
    g2o::SparseOptimizer optimizer; // graph optimizer (why sparse ?)
    optimizer.setAlgorithm(solver); // set algorithm to solver
    optimizer.setVerbose(true);     // set debug output


    // ADD VERTICES & EDGES ===================================================
    // vertex -----------------------------------------------------------------
    Curve_Fitting_Vertex* vertex = new Curve_Fitting_Vertex();
    
    vertex->setId(0); // set as i-th vertex (0-th in this case)
    vertex->setEstimate(Eigen::Vector3d(a_est, b_est, c_est));
    
    optimizer.addVertex(vertex);

    // edge -------------------------------------------------------------------
    for (int i = 0; i < N; i++)
    {
        Curve_Fitting_Edge* edge = new Curve_Fitting_Edge(x_data[i]);

        edge->setId(i);
        edge->setVertex(0, vertex); // set vertex* as i-th vertex (why should this be done (?)) // connect edge to vertex
        edge->setMeasurement(y_data[i]);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma)); // set information matrix

        optimizer.addEdge(edge);
    }
    

    // PERFORM OPTIMIZATION ===================================================
    // time -------------------------------------------------------------------
    std::cout << "start optimization" << std::endl;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    // ------------------------------------------------------------------------
    
    // initiate
    optimizer.initializeOptimization();

    // run
    const int NUM_ITERATIONS = 10;
    optimizer.optimize(NUM_ITERATIONS);

    // ------------------------------------------------------------------------
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "solve time cost = " << time_used.count() << " seconds. " << std::endl;
    // ------------------------------------------------------------------------


    // OPTIMIZATION RESULT ====================================================
    Eigen::Vector3d abc_estimate = vertex->estimate();

    // print ------------------------------------------------------------------
    std::cout << "estimated model: " << abc_estimate.transpose() << std::endl;


    return 0;
}
