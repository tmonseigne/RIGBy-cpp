///-------------------------------------------------------------------------------------------------
/// 
/// \file test_UtilsMedian.hpp
/// \brief Tests for Utils : Median
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 29/07/2020.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// \remarks We use the EEglab Matlab plugin to compare result for validation
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include "gtest/gtest.h"
#include "test_Init.hpp"
#include "test_Misc.hpp"

#include "utils/Median.hpp"
#include "utils/Basics.hpp"

//---------------------------------------------------------------------------------------------------
class Tests_Medians : public testing::Test
{
protected:
	std::vector<Eigen::MatrixXd> m_dataSet;

	void SetUp() override { m_dataSet = Vector2DTo1D(InitCovariance::LWF::Reference()); }
};
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Medians, SimpleMedian)
{
	std::vector<double> v{ 5, 6, 4, 3, 2, 6, 7, 9, 3 };
	double calc = Median(v);
	EXPECT_TRUE(calc == 5);

	v.pop_back();
	calc = Median(v);
	EXPECT_TRUE(calc == 5.5);

	Eigen::MatrixXd m(3, 3);
	m << 5, 6, 4, 3, 2, 6, 7, 9, 3;
	calc = Median(m);
	EXPECT_TRUE(calc == 5);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Medians, DatasetMedian)
{
	Eigen::MatrixXd calc;
	Eigen::MatrixXd ref(3, 3);
	ref << 1.749537973777478, 0.002960131606861, 0.020507254841909,
			0.002960131606861, 1.754563395557952, 0.043042786354499,
			0.020507254841909, 0.043042786354499, 1.057672472691352;
	EXPECT_TRUE(Median(m_dataSet, calc)) << "Error During Median Computes";
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Median of Dataset", ref, calc);
}
//---------------------------------------------------------------------------------------------------
