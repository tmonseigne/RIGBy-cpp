#pragma once

#include "gtest/gtest.h"
#include "test_Misc.hpp"
#include "utils/Featurization.hpp"

using namespace Eigen;
using namespace std;

class Featurization_Tests : public testing::Test {};


TEST_F(Featurization_Tests, GetUpperTriangle_Row)
{
	Initialize();
	RowVectorXd ref(3), calc;

	ref << 8.25, -6.97, 8.25;
	for (int i = 0; i < REPEAT_TEST; ++i) { SqueezeUpperTriangle(M1Cov, calc, true); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("GetUpperTriangle Row", ref, calc).str();
	for (int i = 0; i < REPEAT_TEST; ++i) { Featurization(M1Cov, calc, false, MatrixXd::Identity(2, 2)); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("GetUpperTriangle Row By Featurization", ref, calc).str();
}

TEST_F(Featurization_Tests, GetUpperTriangle_Diagonal)
{
	Initialize();
	RowVectorXd ref(3), calc;

	ref << 8.25, 8.25, -6.97;
	for (int i = 0; i < REPEAT_TEST; ++i) { SqueezeUpperTriangle(M1Cov, calc, false); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("GetUpperTriangle Diagonal", ref, calc).str();
	for (int i = 0; i < REPEAT_TEST; ++i) { Featurization(M1Cov, calc, false, MatrixXd::Identity(2, 2)); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("GetUpperTriangle By Featurization", ref, calc).str();
}


TEST_F(Featurization_Tests, Vector2UpperTriangle_Row)
{
	Initialize();
	RowVectorXd base(3);
	MatrixXd ref(2, 2), calc;
	base << 1, 2, 3;
	ref << 1, 2, 0, 3;
	for (int i = 0; i < REPEAT_TEST; ++i) { Vector2UpperTriangle(base, calc, true); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Vector2UpperTriangle_Row", ref, calc).str();
}

TEST_F(Featurization_Tests, Vector2UpperTriangle_Diagonal)
{
	Initialize();
	RowVectorXd base(3);
	MatrixXd ref(2, 2), calc;
	base << 1, 2, 3;
	ref << 1, 3, 0, 2;
	for (int i = 0; i < REPEAT_TEST; ++i) { Vector2UpperTriangle(base, calc, false); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Vector2UpperTriangle Diagonal", ref, calc).str();
}


TEST_F(Featurization_Tests, TangentSpace)
{
	Initialize();
	RowVectorXd ref(3), calc;
	const MatrixXd iN = MatrixXd::Identity(2, 2);

	ref << 1.48473522, -1.75061981, 1.48473522;
	for (int i = 0; i < REPEAT_TEST; ++i) { TangentSpace(M1Cov, calc, iN); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("TangentSpace M1", ref, calc).str();
	for (int i = 0; i < REPEAT_TEST; ++i) { Featurization(M1Cov, calc, true, iN); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("TangentSpace M1 By Featurization", ref, calc).str();

	ref << 1.48473522, 1.75061981, 1.48473522;
	for (int i = 0; i < REPEAT_TEST; ++i) { TangentSpace(M2Cov, calc, iN); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("TangentSpace M2", ref, calc).str();
	for (int i = 0; i < REPEAT_TEST; ++i) { Featurization(M2Cov, calc, true, iN); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("TangentSpace M2 By Featurization", ref, calc).str();
}

TEST_F(Featurization_Tests, UnTangentSpace)
{
	Initialize();
	MatrixXd ref(2, 2), calc;
	const MatrixXd iN = MatrixXd::Identity(2, 2);

	ref << 8.25, -6.97, -6.97, 8.25;
	for (int i = 0; i < REPEAT_TEST; ++i) { UnTangentSpace(M1Tan, calc, iN); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("UnTangentSpace M1", ref, calc).str();

	ref << 8.25, 6.97, 6.97, 8.25;
	for (int i = 0; i < REPEAT_TEST; ++i) { UnTangentSpace(M2Tan, calc, iN); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("UnTangentSpace M2", ref, calc).str();
}
