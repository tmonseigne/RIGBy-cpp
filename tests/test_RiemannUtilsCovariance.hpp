#pragma once

#include "gtest/gtest.h"
#include "test_Misc.hpp"
#include "utils/Covariance.hpp"

using namespace Eigen;
using namespace std;


class Covariances_Tests : public testing::Test {};

TEST_F(Covariances_Tests, Variance_V1)
{
	Initialize();
	double calc = 0;
	const double ref = 8.25;

	for (int i = 0; i < REPEAT_TEST; ++i) { calc = Variance(V1); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Variance V1", ref, calc).str();
}

TEST_F(Covariances_Tests, Covariance)
{
	Initialize();
	double calc = 0;

	double ref = -8.25;
	for (int i = 0; i < REPEAT_TEST; ++i) { calc = Covariance(M1.row(0), M1.row(1)); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Covariance M1", ref, calc).str();

	ref = 8.25;
	for (int i = 0; i < REPEAT_TEST; ++i) { calc = Covariance(M2.row(0), M2.row(1)); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Covariance M2", ref, calc).str();
}

TEST_F(Covariances_Tests, Covariance_Matrix_COV)
{
	Initialize();
	MatrixXd ref(2, 2), calc;

	ref << 8.25, -8.25, -8.25, 8.25;
	for (int i = 0; i < REPEAT_TEST; ++i) { CovarianceMatrix(M1, calc, Estimator_COV, true); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Covariance Matrix COV M1", ref, calc).str();

	ref << 8.25, 8.25, 8.25, 8.25;
	for (int i = 0; i < REPEAT_TEST; ++i) { CovarianceMatrix(M2, calc, Estimator_COV, true); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Covariance Matrix COV M2", ref, calc).str();
}

TEST_F(Covariances_Tests, Covariance_Matrix_SCM)
{
	Initialize();
	MatrixXd ref(2, 2), calc;

	ref << 8.25, -8.25, -8.25, 8.25;
	for (int i = 0; i < REPEAT_TEST; ++i) { CovarianceMatrix(M1, calc, Estimator_SCM, true); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Covariance Matrix SCM M1", ref, calc).str();

	ref << 8.25, 8.25, 8.25, 8.25;
	for (int i = 0; i < REPEAT_TEST; ++i) { CovarianceMatrix(M2, calc, Estimator_SCM, true); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Covariance Matrix SCM M2", ref, calc).str();
}

TEST_F(Covariances_Tests, Covariance_Matrix_LWF)
{
	Initialize();
	MatrixXd ref(2, 2), calc;

	ref << 8.25, -6.97, -6.97, 8.25;
	for (int i = 0; i < REPEAT_TEST; ++i) { CovarianceMatrix(M1, calc, Estimator_LWF, true); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Covariance Matrix LWF M1", ref, calc).str();

	ref << 8.25, 6.97, 6.97, 8.25;
	for (int i = 0; i < REPEAT_TEST; ++i) { CovarianceMatrix(M2, calc, Estimator_LWF, true); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Covariance Matrix LWF M2", ref, calc).str();
}


TEST_F(Covariances_Tests, Covariance_Matrix_OAS)
{
	Initialize();
	MatrixXd ref(2, 2), calc;

	ref << 8.25, -5.25, -5.25, 8.25;
	for (int i = 0; i < REPEAT_TEST; ++i) { CovarianceMatrix(M1, calc, Estimator_OAS, true); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Covariance Matrix OAS M1", ref, calc).str();

	ref << 8.25, 5.25, 5.25, 8.25;
	for (int i = 0; i < REPEAT_TEST; ++i) { CovarianceMatrix(M2, calc, Estimator_OAS, true); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Covariance Matrix OAS M2", ref, calc).str();
}

TEST_F(Covariances_Tests, Covariance_Matrix_MCD)
{
	Initialize();
	MatrixXd ref(2, 2), calc;

	ref << 4, -4, -4, 4;
	for (int i = 0; i < REPEAT_TEST; ++i) { CovarianceMatrix(M1, calc, Estimator_MCD, true); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Covariance Matrix MCD M1", ref, calc).str();

	ref << 4, 4, 4, 4;
	for (int i = 0; i < REPEAT_TEST; ++i) { CovarianceMatrix(M2, calc, Estimator_MCD, true); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Covariance Matrix MCD M2", ref, calc).str();
}


TEST_F(Covariances_Tests, Covariance_Matrix_COR)
{
	Initialize();
	MatrixXd ref(2, 2), calc;

	ref << 1, -1, -1, 1;
	for (int i = 0; i < REPEAT_TEST; ++i) { CovarianceMatrix(M1, calc, Estimator_COR, true); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Covariance Matrix COR M1", ref, calc).str();

	ref << 1, 1, 1, 1;
	for (int i = 0; i < REPEAT_TEST; ++i) { CovarianceMatrix(M2, calc, Estimator_COR, true); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Covariance Matrix COR M2", ref, calc).str();
}
