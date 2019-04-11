///-------------------------------------------------------------------------------------------------
/// 
/// \file test_UtilsBasics.hpp
/// \brief Tests for Riemannian Geometry Utils : Basics
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 09/01/2019.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include "gtest/gtest.h"
#include "test_Init.hpp"
#include "test_Misc.hpp"
#include "utils/Basics.hpp"
#include "utils/Metrics.hpp"

//---------------------------------------------------------------------------------------------------
class Tests_Basics : public testing::Test
{
protected:
	std::vector<std::vector<Eigen::MatrixXd>> m_dataSet;

	void SetUp() override { m_dataSet = InitDataset::Dataset(); }
};
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Basics, MatrixStandardization)
{
	std::vector<std::vector<Eigen::MatrixXd>> calcC, refC = InitBasics::Center::Reference();
	std::vector<std::vector<Eigen::MatrixXd>> calcS, refS = InitBasics::StandardScaler::Reference();
	calcC.resize(m_dataSet.size());
	calcS.resize(m_dataSet.size());
	for (size_t k = 0; k < m_dataSet.size(); ++k)
	{
		calcC[k].resize(m_dataSet[k].size());
		calcS[k].resize(m_dataSet[k].size());
		for (size_t i = 0; i < m_dataSet[k].size(); ++i)
		{
			EXPECT_TRUE(MatrixStandardization(m_dataSet[k][i], calcC[k][i], Standardization_Center)) << "Error During Centerization" << std::endl;
			EXPECT_TRUE(MatrixStandardization(m_dataSet[k][i], calcS[k][i], Standardization_StandardScale)) << "Error During Standard Scaler" << std::endl;
			const std::string title = "Matrix Center Sample [" + std::to_string(k) + "][" + std::to_string(i) + "]";
			EXPECT_TRUE(isAlmostEqual(refC[k][i], calcC[k][i])) << ErrorMsg(title, refC[k][i], calcC[k][i]).str();
			EXPECT_TRUE(isAlmostEqual(refS[k][i], calcS[k][i])) << ErrorMsg(title, refS[k][i], calcS[k][i]).str();
		}
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Basics, GetElements)
{
	Eigen::RowVectorXd ref(3);
	const std::vector<size_t> idx{ 0, 4, 7 };
	ref << -3, -6, -1;
	const Eigen::RowVectorXd calc = GetElements(m_dataSet[0][0].row(0), idx);
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("GetElements", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Basics, ARange)
{
	const std::vector<size_t> ref{ 1, 3, 5, 7, 9 },
							  calc = ARange(size_t(1), size_t(10), size_t(2));
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("ARange", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Basics, Vector2DTo1D)
{
	std::vector<Eigen::MatrixXd> calc = Vector2DTo1D(m_dataSet);
	bool egal = true;
	size_t idx = 0;
	for (auto& set : m_dataSet)
	{
		for (const auto& data : set)
		{
			if (!isAlmostEqual(data, calc[idx++])) { egal = false; }
		}
	}
	EXPECT_TRUE(egal) << "Vector2DTo1D fail";
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Basics, Metrics)
{
	EXPECT_TRUE(MetricToString(Metric_Riemann) == "Riemann");
	EXPECT_TRUE(MetricToString(Metric_Euclidian) == "Euclidian");
	EXPECT_TRUE(MetricToString(Metric_LogEuclidian) == "Log Euclidian");
	EXPECT_TRUE(MetricToString(Metric_LogDet) == "Log Determinant");
	EXPECT_TRUE(MetricToString(Metric_Kullback) == "Kullback");
	EXPECT_TRUE(MetricToString(Metric_ALE) == "AJD-based log-Euclidean");
	EXPECT_TRUE(MetricToString(Metric_Harmonic) == "Harmonic");
	EXPECT_TRUE(MetricToString(Metric_Wasserstein) == "Wasserstein");
	EXPECT_TRUE(MetricToString(Metric_Identity) == "Identity");
	EXPECT_TRUE(StringToMetric("Riemann") == Metric_Riemann);
	EXPECT_TRUE(StringToMetric("Euclidian") == Metric_Euclidian);
	EXPECT_TRUE(StringToMetric("Log Euclidian") == Metric_LogEuclidian);
	EXPECT_TRUE(StringToMetric("Log Determinant") == Metric_LogDet);
	EXPECT_TRUE(StringToMetric("Kullback") == Metric_Kullback);
	EXPECT_TRUE(StringToMetric("AJD-based log-Euclidean") == Metric_ALE);
	EXPECT_TRUE(StringToMetric("Harmonic") == Metric_Harmonic);
	EXPECT_TRUE(StringToMetric("Wasserstein") == Metric_Wasserstein);
	EXPECT_TRUE(StringToMetric("Identity") == Metric_Identity);
	EXPECT_TRUE(StringToMetric("") == Metric_Identity);
}
//---------------------------------------------------------------------------------------------------
