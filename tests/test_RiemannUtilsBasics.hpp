#pragma once

#include "gtest/gtest.h"
#include "test_Misc.hpp"
#include "utils/Basics.hpp"

using namespace Eigen;
using namespace std;

class Tests_Basics : public testing::Test
{
	void SetUp() override { Initialize(); }
};


TEST_F(Tests_Basics, VectorCenter)
{
	RowVectorXd ref(10), calc;

	ref << -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5;
	for (int i = 0; i < REPEAT_TEST; ++i) { VectorCenter(V1, calc); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("VectorCenter V1", ref, calc).str();
}

TEST_F(Tests_Basics, MatrixCenter)
{
	MatrixXd ref(2, 10), calc;

	ref << -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5,
		4.5, 3.5, 2.5, 1.5, 0.5, -0.5, -1.5, -2.5, -3.5, -4.5;
	for (int i = 0; i < REPEAT_TEST; ++i) { MatrixCenter(M1, calc); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("MatrixCenter M1", ref, calc).str();

	ref << -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5,
		-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5;
	for (int i = 0; i < REPEAT_TEST; ++i) { MatrixCenter(M2, calc); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("MatrixCenter M2", ref, calc).str();
}


TEST_F(Tests_Basics, GetElements)
{
	const std::vector<size_t> idx{ 0, 4, 7 };
	RowVectorXd ref(3), calc;

	ref << 1, 5, 8;
	for (int i = 0; i < REPEAT_TEST; ++i) { calc = GetElements(M1.row(0), idx); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("GetElements", ref, calc).str();
}


TEST_F(Tests_Basics, ARange)
{
	const std::vector<size_t> ref{ 1, 3, 5, 7, 9 };

	std::vector<size_t> calc;
	for (int i = 0; i < REPEAT_TEST; ++i) { calc = ARange(size_t(1), size_t(10), size_t(2)); }
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("ARange", ref, calc).str();
}

