///-------------------------------------------------------------------------------------------------
/// 
/// \file test_UtilsGeodesics.hpp
/// \brief Tests for Riemannian Geometry Utils : Geodesics
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

#include <geometry/Geodesic.hpp>

//---------------------------------------------------------------------------------------------------
class Tests_Geodesic : public testing::Test
{
protected:
	std::vector<Eigen::MatrixXd> m_dataSet;

	void SetUp() override { m_dataSet = Geometry::Vector2DTo1D(InitCovariance::LWF::Reference()); }
};

//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Geodesic, Euclidian)
{
	const std::vector<Eigen::MatrixXd> ref = InitGeodesics::Euclidian::Reference();
	const Eigen::MatrixXd mean             = InitMeans::Euclidian::Reference();
	for (size_t i = 0; i < m_dataSet.size(); ++i)
	{
		Eigen::MatrixXd calc;
		Geodesic(mean, m_dataSet[i], calc, Geometry::EMetric::Euclidian, 0.5);
		EXPECT_TRUE(isAlmostEqual(ref[i], calc)) << ErrorMsg("Geodesic Euclidian Sample [" + std::to_string(i) + "]", ref[i], calc);
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Geodesic, LogEuclidian)
{
	const std::vector<Eigen::MatrixXd> ref = InitGeodesics::LogEuclidian::Reference();
	const Eigen::MatrixXd mean             = InitMeans::LogEuclidian::Reference();
	for (size_t i = 0; i < m_dataSet.size(); ++i)
	{
		Eigen::MatrixXd calc;
		Geodesic(mean, m_dataSet[i], calc, Geometry::EMetric::LogEuclidian, 0.5);
		EXPECT_TRUE(isAlmostEqual(ref[i], calc)) << ErrorMsg("Geodesic LogEuclidian Sample [" + std::to_string(i) + "]", ref[i], calc);
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Geodesic, Riemann)
{
	const std::vector<Eigen::MatrixXd> ref = InitGeodesics::Riemann::Reference();
	const Eigen::MatrixXd mean             = InitMeans::Riemann::Reference();
	for (size_t i = 0; i < m_dataSet.size(); ++i)
	{
		Eigen::MatrixXd calc;
		Geodesic(mean, m_dataSet[i], calc, Geometry::EMetric::Riemann, 0.5);
		EXPECT_TRUE(isAlmostEqual(ref[i], calc)) << ErrorMsg("Geodesic Riemann Sample [" + std::to_string(i) + "]", ref[i], calc);
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Geodesic, Identity)
{
	const Eigen::MatrixXd mean = InitMeans::Riemann::Reference(), ref = Eigen::MatrixXd::Identity(NB_CHAN, NB_CHAN);
	for (size_t i = 0; i < m_dataSet.size(); ++i)
	{
		Eigen::MatrixXd calc;
		Geodesic(mean, m_dataSet[i], calc, Geometry::EMetric::Identity, 0.5);
		EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Geodesic Identity Sample [" + std::to_string(i) + "]", ref, calc);
	}
}
//---------------------------------------------------------------------------------------------------
