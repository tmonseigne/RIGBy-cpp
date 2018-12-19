#pragma once

#include "gtest/gtest.h"
#include "test_Misc.hpp"
#include "test_Init.hpp"
#include "utils/Geodesic.hpp"

class Tests_Geodesic : public testing::Test
{
protected:
	std::vector<Eigen::MatrixXd> m_dataSet;

	void SetUp() override { m_dataSet = Vector2DTo1D(InitCovariance::LWF::Dataset()); }
};

TEST_F(Tests_Geodesic, Euclidian)
{
	const std::vector<Eigen::MatrixXd> ref = InitGeodesics::Euclidian::Dataset();
	const Eigen::MatrixXd mean = InitMeans::Euclidian::Dataset();
	for (size_t i = 0; i < m_dataSet.size(); ++i)
	{
		Eigen::MatrixXd calc;
		Geodesic(mean, m_dataSet[i], calc, Metric_Euclidian, 0.5);
		EXPECT_TRUE(isAlmostEqual(ref[i], calc)) << ErrorMsg("Geodesic Euclidian Sample [" + std::to_string(i) + "]", ref[i], calc).str();
	}
}

TEST_F(Tests_Geodesic, LogEuclidian)
{
	const std::vector<Eigen::MatrixXd> ref = InitGeodesics::LogEuclidian::Dataset();
	const Eigen::MatrixXd mean = InitMeans::LogEuclidian::Dataset();
	for (size_t i = 0; i < m_dataSet.size(); ++i)
	{
		Eigen::MatrixXd calc;
		Geodesic(mean, m_dataSet[i], calc, Metric_LogEuclidian, 0.5);
		EXPECT_TRUE(isAlmostEqual(ref[i], calc)) << ErrorMsg("Geodesic LogEuclidian Sample [" + std::to_string(i) + "]", ref[i], calc).str();
	}
}

TEST_F(Tests_Geodesic, Riemann)
{
	const std::vector<Eigen::MatrixXd> ref = InitGeodesics::Riemann::Dataset();
	const Eigen::MatrixXd mean = InitMeans::Riemann::Dataset();
	for (size_t i = 0; i < m_dataSet.size(); ++i)
	{
		Eigen::MatrixXd calc;
		Geodesic(mean, m_dataSet[i], calc, Metric_Riemann, 0.5);
		EXPECT_TRUE(isAlmostEqual(ref[i], calc)) << ErrorMsg("Geodesic Riemann Sample [" + std::to_string(i) + "]", ref[i], calc).str();
	}
}
