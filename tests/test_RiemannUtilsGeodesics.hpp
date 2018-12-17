#pragma once

#include "gtest/gtest.h"
#include "test_Misc.hpp"
#include "utils/Geodesic.hpp"

using namespace Eigen;
using namespace std;

class Tests_Geodesic : public testing::Test
{
	void SetUp() override { Initialize(); }
};

TEST_F(Tests_Geodesic, Riemann)
{
	MatrixXd ref(2, 2), calc;
	ref << 4.41379655, 8.26508696e-15, 7.44458587e-15, 4.41379655;
	for (int i = 0; i < REPEAT_TEST; ++i) { Geodesic(M1Cov, M2Cov, calc, Metric_Riemann, 0.5); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Geodesic Riemann", ref, calc).str();
}

TEST_F(Tests_Geodesic, Euclidian)
{
	MatrixXd ref(2, 2), calc;
	ref << 8.25, 0.0, 0.0, 8.25;
	for (int i = 0; i < REPEAT_TEST; ++i) { Geodesic(M1Cov, M2Cov, calc, Metric_Euclidian, 0.5); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Geodesic Euclidian", ref, calc).str();
}

TEST_F(Tests_Geodesic, LogEuclidian)
{
	MatrixXd ref(2, 2), calc;
	ref << 4.41379655, 0.0, 0.0, 4.41379655;
	for (int i = 0; i < REPEAT_TEST; ++i) { Geodesic(M1Cov, M2Cov, calc, Metric_LogEuclidian, 0.5); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Geodesic LogEuclidian", ref, calc).str();
}
