///-------------------------------------------------------------------------------------------------
/// 
/// \file test_UtilsDistance.hpp
/// \brief Tests for Riemannian Geometry Utils : Distance
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 09/01/2019.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include "gtest/gtest.h"
#include "misc.hpp"
#include "Init.hpp"

#include <geometry/Distance.hpp>

//---------------------------------------------------------------------------------------------------
class Tests_Distances : public testing::Test
{
protected:
	std::vector<Eigen::MatrixXd> m_dataSet;

	void SetUp() override { m_dataSet = Geometry::Vector2DTo1D(InitCovariance::LWF::Reference()); }
};

//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Distances, Euclidian)
{
	const std::vector<double> ref = InitDistance::Euclidian::Reference();
	const Eigen::MatrixXd mean    = InitMeans::Euclidian::Reference();
	for (size_t i = 0; i < m_dataSet.size(); ++i)
	{
		const double calc = Distance(mean, m_dataSet[i], Geometry::EMetric::Euclidian);
		EXPECT_TRUE(isAlmostEqual(ref[i], calc)) << ErrorMsg("Distance Euclidian Sample [" + std::to_string(i) + "]", ref[i], calc);
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Distances, LogEuclidian)
{
	const std::vector<double> ref = InitDistance::LogEuclidian::Reference();
	const Eigen::MatrixXd mean    = InitMeans::LogEuclidian::Reference();
	for (size_t i = 0; i < m_dataSet.size(); ++i)
	{
		const double calc = Distance(mean, m_dataSet[i], Geometry::EMetric::LogEuclidian);
		EXPECT_TRUE(isAlmostEqual(ref[i], calc)) << ErrorMsg("Distance LogEuclidian Sample [" + std::to_string(i) + "]", ref[i], calc);
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Distances, Riemann)
{
	const std::vector<double> ref = InitDistance::Riemann::Reference();
	const Eigen::MatrixXd mean    = InitMeans::Riemann::Reference();
	for (size_t i = 0; i < m_dataSet.size(); ++i)
	{
		const double calc = Distance(mean, m_dataSet[i], Geometry::EMetric::Riemann);
		EXPECT_TRUE(isAlmostEqual(ref[i], calc)) << ErrorMsg("Distance Riemann Sample [" + std::to_string(i) + "]", ref[i], calc);
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Distances, LogDet)
{
	const std::vector<double> ref = InitDistance::LogDeterminant::Reference();
	const Eigen::MatrixXd mean    = InitMeans::LogDeterminant::Reference();
	for (size_t i = 0; i < m_dataSet.size(); ++i)
	{
		const double calc = Distance(mean, m_dataSet[i], Geometry::EMetric::LogDet);
		EXPECT_TRUE(isAlmostEqual(ref[i], calc)) << ErrorMsg("Distance LogDet Sample [" + std::to_string(i) + "]", ref[i], calc);
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Distances, Kullback)
{
	const std::vector<double> ref = InitDistance::Kullback::Reference();
	const Eigen::MatrixXd mean    = InitMeans::Kullback::Reference();
	for (size_t i = 0; i < m_dataSet.size(); ++i)
	{
		const double calc = Distance(mean, m_dataSet[i], Geometry::EMetric::Kullback);
		EXPECT_TRUE(isAlmostEqual(ref[i], calc)) << ErrorMsg("Distance Kullback Sample [" + std::to_string(i) + "]", ref[i], calc);
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Distances, Wasserstein)
{
	const std::vector<double> ref = InitDistance::Wasserstein::Reference();
	const Eigen::MatrixXd mean    = InitMeans::Wasserstein::Reference();
	for (size_t i = 0; i < m_dataSet.size(); ++i)
	{
		const double calc = Distance(mean, m_dataSet[i], Geometry::EMetric::Wasserstein);
		EXPECT_TRUE(isAlmostEqual(ref[i], calc)) << ErrorMsg("Distance Wasserstein Sample [" + std::to_string(i) + "]", ref[i], calc);
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Distances, Identity)
{
	const Eigen::MatrixXd mean = InitMeans::Wasserstein::Reference();
	for (size_t i = 0; i < m_dataSet.size(); ++i)
	{
		const double calc = Distance(mean, m_dataSet[i], Geometry::EMetric::Identity);
		EXPECT_TRUE(isAlmostEqual(1, calc)) << ErrorMsg("Distance Wasserstein Sample [" + std::to_string(i) + "]", 1, calc);
	}
}
//---------------------------------------------------------------------------------------------------
