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
			EXPECT_TRUE(MatrixStandardization(m_dataSet[k][i], calcC[k][i], EStandardization::Center)) << "Error During Centerization" << std::endl;
			EXPECT_TRUE(MatrixStandardization(m_dataSet[k][i], calcS[k][i], EStandardization::StandardScale)) << "Error During Standard Scaler" << std::endl;
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
	bool egal                         = true;
	size_t idx                        = 0;
	for (auto& set : m_dataSet) { for (const auto& data : set) { if (!isAlmostEqual(data, calc[idx++])) { egal = false; } } }
	EXPECT_TRUE(egal) << "Vector2DTo1D fail";
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Basics, Metrics)
{
	EXPECT_TRUE(toString(EMetric::Riemann) == "Riemann");
	EXPECT_TRUE(toString(EMetric::Euclidian) == "Euclidian");
	EXPECT_TRUE(toString(EMetric::LogEuclidian) == "Log Euclidian");
	EXPECT_TRUE(toString(EMetric::LogDet) == "Log Determinant");
	EXPECT_TRUE(toString(EMetric::Kullback) == "Kullback");
	EXPECT_TRUE(toString(EMetric::ALE) == "AJD-based log-Euclidean");
	EXPECT_TRUE(toString(EMetric::Harmonic) == "Harmonic");
	EXPECT_TRUE(toString(EMetric::Wasserstein) == "Wasserstein");
	EXPECT_TRUE(toString(EMetric::Identity) == "Identity");
	EXPECT_TRUE(StringToMetric("Riemann") == EMetric::Riemann);
	EXPECT_TRUE(StringToMetric("Euclidian") == EMetric::Euclidian);
	EXPECT_TRUE(StringToMetric("Log Euclidian") == EMetric::LogEuclidian);
	EXPECT_TRUE(StringToMetric("Log Determinant") == EMetric::LogDet);
	EXPECT_TRUE(StringToMetric("Kullback") == EMetric::Kullback);
	EXPECT_TRUE(StringToMetric("AJD-based log-Euclidean") == EMetric::ALE);
	EXPECT_TRUE(StringToMetric("Harmonic") == EMetric::Harmonic);
	EXPECT_TRUE(StringToMetric("Wasserstein") == EMetric::Wasserstein);
	EXPECT_TRUE(StringToMetric("Identity") == EMetric::Identity);
	EXPECT_TRUE(StringToMetric("") == EMetric::Identity);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Basics, Validation)
{
	const Eigen::MatrixXd m1 = Eigen::MatrixXd::Zero(2, 2),
						  m2 = Eigen::MatrixXd::Zero(1, 2),
						  m3;
	std::vector<Eigen::MatrixXd> v;

	EXPECT_TRUE(InRange(1, 0, 2));
	EXPECT_FALSE(InRange(2, 0, 1));

	EXPECT_FALSE(AreNotEmpty(v));
	v.push_back(m3);
	EXPECT_FALSE(AreNotEmpty(v));
	v.push_back(m1);
	EXPECT_FALSE(AreNotEmpty(v));
	v.clear();
	v.push_back(m1);
	v.push_back(m2);
	EXPECT_TRUE(AreNotEmpty(v));

	EXPECT_TRUE(HaveSameSize(m1, m1));
	EXPECT_FALSE(HaveSameSize(m3, m3) && HaveSameSize(m1, m2) && HaveSameSize(m1, m3));

	EXPECT_FALSE(HaveSameSize(v) && AreSquare(v));
	v.clear();
	v.push_back(m1);
	v.push_back(m1);
	EXPECT_TRUE(HaveSameSize(v) && AreSquare(v));

	MatrixPrint(m1);
	MatrixPrint(m3);

	std::vector<std::string> vs = Split("0,1,2,3.a\n", ",");
	EXPECT_TRUE(vs.size() == 4 && vs[0] == "0" && vs[1] == "1" && vs[2] == "2" && vs[3] == "3.a") << vs.size() << " " << vs[0] << " " << vs[1] << " " << vs[2]
 << " " << vs[3] << std::endl;
}
//---------------------------------------------------------------------------------------------------
