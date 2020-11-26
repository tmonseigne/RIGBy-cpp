///-------------------------------------------------------------------------------------------------
/// 
/// \file test_Median.hpp
/// \brief Tests for Utils : Misc
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 29/07/2020.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// \remarks We use the EEglab Matlab plugin to compare result for validation
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include "gtest/gtest.h"

#include "Init.hpp"
#include "misc.hpp"

#include <geometry/Basics.hpp>
#include <geometry/Median.hpp>

//---------------------------------------------------------------------------------------------------
class Tests_Median : public testing::Test
{
protected:
	std::vector<Eigen::MatrixXd> m_dataSet;

	void SetUp() override { m_dataSet = Geometry::Vector2DTo1D(InitCovariance::LWF::Reference()); }
};

//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Median, SimpleMedian)
{
	std::vector<double> v{ 5, 6, 4, 3, 2, 6, 7, 9, 3 };
	double calc = Geometry::Median(v);
	EXPECT_EQ(calc, 5);

	v.pop_back();
	calc = Geometry::Median(v);
	EXPECT_EQ(calc, 5.5);

	Eigen::MatrixXd m(3, 3);
	m << 5, 6, 4, 3, 2, 6, 7, 9, 3;
	calc = Geometry::Median(m);
	EXPECT_EQ(calc, 5);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Median, DatasetEuclidianMedian)
{
	Eigen::MatrixXd calc;
	Eigen::MatrixXd ref(3, 3);
	ref << 1.749537973777478, 0.002960131606861, 0.020507254841909,
			0.002960131606861, 1.754563395557952, 0.043042786354499,
			0.020507254841909, 0.043042786354499, 1.057672472691352;
	EXPECT_TRUE(Geometry::Median(m_dataSet, calc, 0.0001, 50, Geometry::EMetric::Euclidian)) << "Error During Median Computes";
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Median of Dataset", ref, calc);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Median, DatasetRiemannMedian)
{
	Eigen::MatrixXd calc;
	Eigen::MatrixXd ref(3, 3);
	ref << 1.851330747504982, 0.002002346316770, 0.022122030618131,
			0.002002346316770, 1.644242996651016, 0.033655563302757,
			0.022122030618131, 0.033655563302757, 0.851184143800763;
	EXPECT_TRUE(Geometry::Median(m_dataSet, calc, 0.0001, 50, Geometry::EMetric::Riemann)) << "Error During Median Computes";
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Median of Dataset", ref, calc);
}
//---------------------------------------------------------------------------------------------------
