#pragma once

#include "gtest/gtest.h"
#include "test_Misc.hpp"
#include "utils/Distance.hpp"

using namespace Eigen;
using namespace std;

class Tests_Distances : public testing::Test
{
	void SetUp() override { Initialize(); }
};

TEST_F(Tests_Distances, Euclidian)
{
	double calc = 0;

	const double ref = 19.714137059480944;
	for (int i = 0; i < REPEAT_TEST; ++i) { calc = Distance(M1Cov, M2Cov, Metric_Euclidian); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Distance Euclidian", ref, calc).str();
}

TEST_F(Tests_Distances, LogEuclidian)
{
	double calc = 0;

	const double ref = 3.501239615249661;
	for (int i = 0; i < REPEAT_TEST; ++i) { calc = Distance(M1Cov, M2Cov, Metric_LogEuclidian); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Distance LogEuclidian", ref, calc).str();
}

TEST_F(Tests_Distances, Riemann)
{
	double calc = 0;

	const double ref = 3.5012396152496637;
	for (int i = 0; i < REPEAT_TEST; ++i) { calc = Distance(M1Cov, M2Cov, Metric_Riemann); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Distance Riemann", ref, calc).str();
}

TEST_F(Tests_Distances, LogDet)
{
	double calc = 0;

	const double ref = 1.1184614299689164;
	for (int i = 0; i < REPEAT_TEST; ++i) { calc = Distance(M1Cov, M2Cov, Metric_LogDet); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Distance LogDet", ref, calc).str();
}

TEST_F(Tests_Distances, Kullback)
{
	double calc = 0;

	const double ref = 9.974724868593952;
	for (int i = 0; i < REPEAT_TEST; ++i) { calc = Distance(M1Cov, M2Cov, Metric_Kullback); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Distance Kullback", ref, calc).str();
}

TEST_F(Tests_Distances, Wasserstein)
{
	double calc = 0;

	const double ref = 3.917245689653489;
	for (int i = 0; i < REPEAT_TEST; ++i) { calc = Distance(M1Cov, M2Cov, Metric_Wasserstein); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Distance Wasserstein", ref, calc).str();
}
