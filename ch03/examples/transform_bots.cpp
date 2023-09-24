// transform_bots.cpp
// 2023 APR 24
// Tershire

// HEADER FILE ////////////////////////////////////////////////////////////////
#include <iostream>
#include <vector>
#include <algorithm>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

// MAIN ///////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    using namespace std;
    using namespace Eigen;

    // SETUP ==================================================================
    // setup & normalize quaternion
    Quaterniond q1(0.35, 0.2, 0.3, 0.1), q2(-0.5, 0.4, -0.1, 0.2);
    q1.normalize();
    q2.normalize();

    // setup translation vector
    Vector3d t1(0.3, 0.1, 0.1), t2(-0.1, 0.5, 0.3);

    // setup observation vector
    Vector3d r_p_wrt_B1_B1(0.5, 0, 0.2);
    cout << "r_p_wrt_B1_B1:\n" << r_p_wrt_B1_B1.transpose() << endl;

    // TRANSFORM MATRIX =======================================================
    Isometry3d E_B1_W(q1), E_B2_W(q2);
    // E_B1_W.prerotate(q1);
    E_B1_W.pretranslate(t1);
    // E_B2_W.prerotate(q2);
    E_B2_W.pretranslate(t2);

    // TRANSFORM ==============================================================
    Vector3d r_p_wrt_B1_B2 = E_B2_W * E_B1_W.inverse() * r_p_wrt_B1_B1;
    cout << "r_p_wrt_B1_B2:\n" << r_p_wrt_B1_B2.transpose() << endl;

    return 0;
}
