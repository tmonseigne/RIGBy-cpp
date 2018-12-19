#pragma once

#include "gtest/gtest.h"
#include "test_Misc.hpp"
#include "test_Init.hpp"
#include "utils/Mean.hpp"

class Tests_Means : public testing::Test
{
protected:
	std::vector<Eigen::MatrixXd> m_dataSet;

	void SetUp() override { m_dataSet = Vector2DTo1D(InitCovariance::LWF::Dataset()); }
};

TEST_F(Tests_Means, Euclidian)
{
	Eigen::MatrixXd calc, ref = InitMeans::Euclidian::Dataset();
	Mean(m_dataSet, calc, Metric_Euclidian);
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Mean Matrix Euclidian", ref, calc).str();
}

TEST_F(Tests_Means, LogEuclidian)
{
	Eigen::MatrixXd calc, ref = InitMeans::LogEuclidian::Dataset();
	Mean(m_dataSet, calc, Metric_LogEuclidian);
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Mean Matrix LogEuclidian", ref, calc).str();
}

TEST_F(Tests_Means, Riemann)
{
	Eigen::MatrixXd calc, ref = InitMeans::Riemann::Dataset();
	Mean(m_dataSet, calc, Metric_Riemann);
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Mean Matrix Riemann", ref, calc).str();
}

TEST_F(Tests_Means, LogDet)
{
	Eigen::MatrixXd calc, ref = InitMeans::LogDeterminant::Dataset();
	Mean(m_dataSet, calc, Metric_LogDet);
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Mean Matrix LogDet", ref, calc).str();
}

TEST_F(Tests_Means, Kullback)
{
	Eigen::MatrixXd calc, ref = InitMeans::Kullback::Dataset();
	Mean(m_dataSet, calc, Metric_Kullback);
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Mean Matrix Kullback", ref, calc).str();
}

TEST_F(Tests_Means, Wasserstein)
{
	Eigen::MatrixXd calc, ref = InitMeans::Wasserstein::Dataset();
	Mean(m_dataSet, calc, Metric_Wasserstein);
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Mean Matrix Wasserstein", ref, calc).str();
}

TEST_F(Tests_Means, ALE)
{
	EXPECT_TRUE(false) << "Not implemented";
	/*
	Eigen::MatrixXd calc, ref = InitMeans::ALE::Dataset();
	Mean(m_dataSet, calc, Metric_ALE);
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Mean Matrix ALE", ref, calc).str();
	*/
}

TEST_F(Tests_Means, Harmonic)
{
	Eigen::MatrixXd calc, ref = InitMeans::Harmonic::Dataset();
	Mean(m_dataSet, calc, Metric_Harmonic);
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Mean Matrix Harmonic", ref, calc).str();
}

TEST_F(Tests_Means, Identity)
{
	Eigen::MatrixXd calc, ref = InitMeans::Identity::Dataset();
	Mean(m_dataSet, calc, Metric_Identity);
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Mean Matrix Identity", ref, calc).str();
}
