// eigen_basics.cpp
// 2023 APR 23
// Tershire

// HEADER FILE ////////////////////////////////////////////////////////////////
#include <iostream>
#include <ctime>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

// CONSTANT ///////////////////////////////////////////////////////////////////
const int N = 50;

// MAIN ///////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    using namespace std;
    using namespace Eigen;

    // DECLARATION ============================================================
    // vector
    Vector3d v_3d;
    Matrix<float, 3, 1> v_3f;

    // matrix
    Matrix<float, 2, 3> matrix_2by3;

    // square matrix
    Matrix3d matrix_3by3;
    
    // dynamic size
    Matrix<double, Dynamic, 3> matrix_Xby3;
    MatrixXd matrix_XbyX;

    // ASSIGNMENT =============================================================
    v_3d << 3, 2, 1;
    v_3f << 4, 5, 6;
    
    matrix_2by3 << 1, 2, 3, 4, 5, 6;
    cout << "matrix 2X3:\n" << matrix_2by3 << endl;
    cout << "matrix 2X3:\n";
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            cout << matrix_2by3(i, j) << "\t\n";
        }
    }

    // zero
    matrix_3by3 = Matrix3d::Zero();
    cout << "zero:\n" << matrix_3by3 << endl;

    // random
    matrix_3by3 = Matrix3d::Random();
    cout << "random:\n" << matrix_3by3 << endl;

    // OPERATION ==============================================================
    // BASIC ------------------------------------------------------------------
    // multiplication
    Matrix<float, 2, 1> res_f = matrix_2by3 * v_3f;
    cout << "[1, 2, 3; 4, 5, 6]*[4, 5, 6] =\n" << res_f.transpose() << endl;
    
    Matrix<double, 2, 1> res_d = matrix_2by3.cast<double>() * v_3d; // comply types
    cout << "[1, 2, 3; 4, 5, 6]*[3, 2, 1] =\n" << res_d.transpose() << endl;

    // transpose
    cout << "transpose:\n" << matrix_3by3.transpose() << endl;

    // sum
    cout << "sum:\n" << matrix_3by3.sum() << endl;

    // trace
    cout << "trace:\n" << matrix_3by3.trace() << endl;

    // det
    cout << "det:\n" << matrix_3by3.determinant() << endl;

    // inverse
    cout << "inverse:\n" << matrix_3by3.inverse() << endl;

    // eigen ------------------------------------------------------------------
    Matrix3d matrix_3by3_symmetric = matrix_3by3.transpose() * matrix_3by3;
    SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_3by3_symmetric);
    cout << "Eigen Values :\n" << eigen_solver.eigenvalues() << endl;
    cout << "Eigen Vectors:\n" << eigen_solver.eigenvectors() << endl;

    // solve linear system ----------------------------------------------------
    // matrix_NbyN *x = v_Nd
    Matrix<double, N, N> matrix_NbyN = MatrixXd::Random(N, N); // need to specify (N, N) here
    matrix_NbyN = matrix_NbyN.transpose() * matrix_NbyN;
    Matrix<double, N, 1> v_Nd = MatrixXd::Random(N, 1);

    // setting
    clock_t time_start;
    Matrix<double, N, 1> x;

    // <1> Direct Inversion
    time_start = clock();
    x = matrix_NbyN.inverse() * v_Nd;
    cout << "<1> Direct Inversion:\n"
         << "time: " << (clock() - time_start) * 1000 / (double) CLOCKS_PER_SEC
         << "[ms]\n";
    cout << "ans : " << x.transpose() << endl; 
    
    // <2> QR Decomposition
    time_start = clock();
    x = matrix_NbyN.colPivHouseholderQr().solve(v_Nd);
    cout << "<2> QR Decomposition:\n"
         << "time: " << (clock() - time_start) * 1000 / (double) CLOCKS_PER_SEC
         << "[ms]\n";
    cout << "ans : " << x.transpose() << endl; 

    // <3> Cholesky Decomposition
    time_start = clock();
    x = matrix_NbyN.ldlt().solve(v_Nd);
    cout << "<3> Cholesky Decomposition:\n"
         << "time: " << (clock() - time_start) * 1000 / (double) CLOCKS_PER_SEC
         << "[ms]\n";
    cout << "ans : " << x.transpose() << endl; 

    return 0;
}
