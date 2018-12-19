#pragma once

#include "gtest/gtest.h"
#include "test_Misc.hpp"
#include "test_Init.hpp"
#include "utils/Featurization.hpp"

class Tests_Featurization : public testing::Test 
{
protected:
	std::vector<Eigen::MatrixXd> m_dataSet;

	void SetUp() override { m_dataSet = Vector2DTo1D(InitCovariance::LWF::Dataset()); }
};

TEST_F(Tests_Featurization, TangentSpace)
{
	const std::vector<Eigen::RowVectorXd> ref = InitFeaturization::TangentSpace::Dataset();
	const Eigen::MatrixXd mean = InitMeans::Riemann::Dataset();
	for (size_t i = 0; i < m_dataSet.size(); ++i)
	{
		Eigen::RowVectorXd calc;
		Featurization(m_dataSet[i], calc, true, mean);
		EXPECT_TRUE(isAlmostEqual(ref[i], calc)) << ErrorMsg("TangentSpace Sample [" + std::to_string(i) + "]", ref[i], calc).str();
	}
}

TEST_F(Tests_Featurization, UnTangentSpace)
{
	const std::vector<Eigen::RowVectorXd> ref = InitFeaturization::TangentSpace::Dataset();
	const Eigen::MatrixXd mean = InitMeans::Riemann::Dataset();
	for (size_t i = 0; i < m_dataSet.size(); ++i)
	{
		Eigen::MatrixXd calc;
		UnFeaturization(ref[i], calc, true, mean);
		EXPECT_TRUE(isAlmostEqual(m_dataSet[i], calc)) << ErrorMsg("UnTangentSpace Sample [" + std::to_string(i) + "]", m_dataSet[i], calc).str();
	}
}

TEST_F(Tests_Featurization, Squeeze)
{
	const std::vector<Eigen::RowVectorXd> ref = InitFeaturization::Squeeze::Dataset();
	const Eigen::MatrixXd mean = InitMeans::Riemann::Dataset();
	for (size_t i = 0; i < m_dataSet.size(); ++i)
	{
		Eigen::RowVectorXd calc;
		Featurization(m_dataSet[i], calc, false, mean);
		EXPECT_TRUE(isAlmostEqual(ref[i], calc)) << ErrorMsg("Squeeze Sample [" + std::to_string(i) + "]", ref[i], calc).str();
	}
}

TEST_F(Tests_Featurization, UnSqueeze)
{
	const std::vector<Eigen::RowVectorXd> ref = InitFeaturization::Squeeze::Dataset();
	const Eigen::MatrixXd mean = InitMeans::Riemann::Dataset();
	for (size_t i = 0; i < m_dataSet.size(); ++i)
	{
		Eigen::MatrixXd calc;
		UnFeaturization(ref[i], calc, false, mean);
		EXPECT_TRUE(isAlmostEqual(m_dataSet[i], calc)) << ErrorMsg("UnSqueeze Sample [" + std::to_string(i) + "]", m_dataSet[i], calc).str();
	}
}
