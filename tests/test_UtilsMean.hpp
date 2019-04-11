///-------------------------------------------------------------------------------------------------
/// 
/// \file test_UtilsMean.hpp
/// \brief Tests for Riemannian Geometry Utils : Mean
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 09/01/2019.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include "gtest/gtest.h"
#include "test_Misc.hpp"
#include "test_Init.hpp"
#include "utils/Mean.hpp"

//---------------------------------------------------------------------------------------------------
class Tests_Means : public testing::Test
{
protected:
	std::vector<Eigen::MatrixXd> m_dataSet;

	void SetUp() override { m_dataSet = Vector2DTo1D(InitCovariance::LWF::Reference()); }
};
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Means, BadInput)
{
	std::vector<Eigen::MatrixXd> bad;
	Eigen::MatrixXd calc;
	EXPECT_FALSE(Mean(bad, calc, Metric_Riemann));
	bad.emplace_back(Eigen::MatrixXd::Zero(1, 2));
	bad.emplace_back(Eigen::MatrixXd::Zero(1, 2));
	EXPECT_FALSE(Mean(bad, calc, Metric_Riemann));
	bad.emplace_back(Eigen::MatrixXd::Zero(2, 2));
	EXPECT_FALSE(Mean(bad, calc, Metric_Riemann));
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Means, Euclidian)
{
	const Eigen::MatrixXd ref = InitMeans::Euclidian::Reference();
	Eigen::MatrixXd calc;
	Mean(m_dataSet, calc, Metric_Euclidian);
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Mean Matrix Euclidian", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Means, LogEuclidian)
{
	const Eigen::MatrixXd ref = InitMeans::LogEuclidian::Reference();
	Eigen::MatrixXd calc;
	Mean(m_dataSet, calc, Metric_LogEuclidian);
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Mean Matrix LogEuclidian", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Means, Riemann)
{
	const Eigen::MatrixXd ref = InitMeans::Riemann::Reference();
	Eigen::MatrixXd calc;
	Mean(m_dataSet, calc, Metric_Riemann);
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Mean Matrix Riemann", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Means, LogDet)
{
	const Eigen::MatrixXd ref = InitMeans::LogDeterminant::Reference();
	Eigen::MatrixXd calc;
	Mean(m_dataSet, calc, Metric_LogDet);
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Mean Matrix LogDet", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Means, Kullback)
{
	const Eigen::MatrixXd ref = InitMeans::Kullback::Reference();
	Eigen::MatrixXd calc;
	Mean(m_dataSet, calc, Metric_Kullback);
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Mean Matrix Kullback", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Means, Wasserstein)
{
	std::cout << "Precision Error";
	const Eigen::MatrixXd ref = InitMeans::Wasserstein::Reference();
	Eigen::MatrixXd calc;
	Mean(m_dataSet, calc, Metric_Wasserstein);
	/*
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Mean Matrix Wasserstein", ref, calc).str();
	*/
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Means, ALE)
{
	std::cout << "Not implemented";
	Eigen::MatrixXd calc, ref = InitMeans::ALE::Reference();
	Mean(m_dataSet, calc, Metric_ALE);
	/*
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Mean Matrix ALE", ref, calc).str();
	*/
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Means, Harmonic)
{
	const Eigen::MatrixXd ref = InitMeans::Harmonic::Reference();
	Eigen::MatrixXd calc;
	Mean(m_dataSet, calc, Metric_Harmonic);
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Mean Matrix Harmonic", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Means, Identity)
{
	const Eigen::MatrixXd ref = InitMeans::Identity::Reference();
	Eigen::MatrixXd calc;
	Mean(m_dataSet, calc, Metric_Identity);
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Mean Matrix Identity", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------
