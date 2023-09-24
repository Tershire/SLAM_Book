// eigen_geometry.cpp
// 2023 APR 24
// Tershire

// HEADER FILE ////////////////////////////////////////////////////////////////
#include <iostream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

// CONSTANT ///////////////////////////////////////////////////////////////////
const int N = 50;

// MAIN ///////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    using namespace std;
    using namespace Eigen;

    // ROTATION ===============================================================
    // rotation matrix
    Matrix3d rotation_matrix = Matrix3d::Identity();

    // rotation vector
    AngleAxisd rotation_vector(M_PI / 4, Vector3d(0, 0, 1));

    // convert to rotation matrix
    // <1>
    cout.precision(3);
    cout << "rotation_vector.matrix() =\n" << rotation_vector.matrix() << endl;

    // <2>
    rotation_matrix = rotation_vector.toRotationMatrix();
    cout << "rotation_matrix =\n" << rotation_matrix << endl;

    // COORDINATE TRANSFORM ---------------------------------------------------
    Vector3d v(1, 0, 0);
    cout << "vector to rotate =\n" << v.transpose() << endl;
    Vector3d v_rotated;

    // <1> using rotation vector
    v_rotated = rotation_vector * v;
    cout << "v_rotated (using AngleAxis) =\n" << v_rotated.transpose() << endl;

    // <2> using rotation matrix
    v_rotated = rotation_matrix * v;
    cout << "v_rotated (using AngleAxis) =\n" << v_rotated.transpose() << endl;

    // VARIOUS ATTITUDE SYSTEMS -----------------------------------------------
    // Euler angle
    Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0);
    cout << "yaw, pitch, roll =\n" << euler_angles.transpose() << endl;

    // EUCLIDEAN TRANSFORM ====================================================
    Isometry3d E_hom = Isometry3d::Identity(); // in fact, 4X4 as a result

    // setup {rotation & translation}
    E_hom.rotate(rotation_vector);
    E_hom.pretranslate(Vector3d(1, 3, 4));
    cout << "E =\n" << E_hom.matrix() << endl;

    // COORDINATE TRANSFORM ---------------------------------------------------
    Vector3d v_transformed = E_hom * v;
    cout << "v_transformed =\n" << v_transformed.transpose() << endl;

    // QUATERNION =============================================================
    Quaterniond q;
    // (q0, q1, q2, q3)
    
    // <1> using rotation vector
    q = Quaterniond(rotation_vector);
    cout << "quaternion linked to rotation vector =\n" 
         << q.coeffs().transpose() << endl;

    // <2> using rotation matrix
    q = Quaterniond(rotation_matrix);
    cout << "quaternion linked to rotation matrix =\n"
         << q.coeffs().transpose() << endl;

    // rotation
    v_rotated = q * v;
    cout << "v_rotated =\n" << v_rotated.transpose() << endl;

    // check
    cout << "should be equal to\n"
         << (q * Quaterniond(0, 1, 0, 0) * q.inverse()).coeffs().transpose()
         << endl;

    return 0;
}
