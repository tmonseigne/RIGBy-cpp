#pragma once

#include "gtest/gtest.h"
#include "test_Misc.hpp"
#include "test_Init.hpp"
#include "utils/Distance.hpp"

class Tests_Distances : public testing::Test
{
protected:
	std::vector<Eigen::MatrixXd> m_dataSet;

	void SetUp() override
	{
		m_dataSet = Vector2DTo1D(InitCovariance::LWF::Dataset());
	}
};

TEST_F(Tests_Distances, Euclidian)
{
	const std::vector<double> ref = InitDistance::Euclidian::Dataset();
	const Eigen::MatrixXd mean = InitMeans::Euclidian::Dataset();
	for (size_t i = 0; i < m_dataSet.size(); ++i)
	{
		const double calc = Distance(mean, m_dataSet[i], Metric_Euclidian);
		EXPECT_TRUE(isAlmostEqual(ref[i], calc)) << ErrorMsg("Distance Euclidian Sample [" + std::to_string(i) + "]", ref[i], calc).str();
	}
}

TEST_F(Tests_Distances, LogEuclidian)
{
	const std::vector<double> ref = InitDistance::LogEuclidian::Dataset();
	const Eigen::MatrixXd mean = InitMeans::LogEuclidian::Dataset();
	for (size_t i = 0; i < m_dataSet.size(); ++i)
	{
		const double calc = Distance(mean, m_dataSet[i], Metric_LogEuclidian);
		EXPECT_TRUE(isAlmostEqual(ref[i], calc)) << ErrorMsg("Distance LogEuclidian Sample [" + std::to_string(i) + "]", ref[i], calc).str();
	}
}

TEST_F(Tests_Distances, Riemann)
{
	const std::vector<double> ref = InitDistance::Riemann::Dataset();
	const Eigen::MatrixXd mean = InitMeans::Riemann::Dataset();
	for (size_t i = 0; i < m_dataSet.size(); ++i)
	{
		const double calc = Distance(mean, m_dataSet[i], Metric_Riemann);
		EXPECT_TRUE(isAlmostEqual(ref[i], calc)) << ErrorMsg("Distance Riemann Sample [" + std::to_string(i) + "]", ref[i], calc).str();
	}
}

TEST_F(Tests_Distances, LogDet)
{
	const std::vector<double> ref = InitDistance::LogDeterminant::Dataset();
	const Eigen::MatrixXd mean = InitMeans::LogDeterminant::Dataset();
	for (size_t i = 0; i < m_dataSet.size(); ++i)
	{
		const double calc = Distance(mean, m_dataSet[i], Metric_LogDet);
		EXPECT_TRUE(isAlmostEqual(ref[i], calc)) << ErrorMsg("Distance LogDet Sample [" + std::to_string(i) + "]", ref[i], calc).str();
	}
}

TEST_F(Tests_Distances, Kullback)
{
	const std::vector<double> ref = InitDistance::Kullback::Dataset();
	const Eigen::MatrixXd mean = InitMeans::Kullback::Dataset();
	for (size_t i = 0; i < m_dataSet.size(); ++i)
	{
		const double calc = Distance(mean, m_dataSet[i], Metric_Kullback);
		EXPECT_TRUE(isAlmostEqual(ref[i], calc)) << ErrorMsg("Distance Kullback Sample [" + std::to_string(i) + "]", ref[i], calc).str();
	}
}

TEST_F(Tests_Distances, Wasserstein)
{
	const std::vector<double> ref = InitDistance::Wasserstein::Dataset();
	const Eigen::MatrixXd mean = InitMeans::Wasserstein::Dataset();
	for (size_t i = 0; i < m_dataSet.size(); ++i)
	{
		const double calc = Distance(mean, m_dataSet[i], Metric_Wasserstein);
		EXPECT_TRUE(isAlmostEqual(ref[i], calc)) << ErrorMsg("Distance Wasserstein Sample [" + std::to_string(i) + "]", ref[i], calc).str();
	}
}
