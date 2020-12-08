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
#include "Init.hpp"
#include "misc.hpp"

#include <geometry/Basics.hpp>
#include <geometry/Metrics.hpp>

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
			EXPECT_TRUE(MatrixStandardization(m_dataSet[k][i], calcC[k][i], Geometry::EStandardization::Center)) << "Error During Centerization" << std::endl;
			EXPECT_TRUE(MatrixStandardization(m_dataSet[k][i], calcS[k][i], Geometry::EStandardization::StandardScale)) << "Error During Standard Scaler" << std::endl;
			const std::string title = "Matrix Center Sample [" + std::to_string(k) + "][" + std::to_string(i) + "]";
			EXPECT_TRUE(isAlmostEqual(refC[k][i], calcC[k][i])) << ErrorMsg(title, refC[k][i], calcC[k][i]);
			EXPECT_TRUE(isAlmostEqual(refS[k][i], calcS[k][i])) << ErrorMsg(title, refS[k][i], calcS[k][i]);
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
	const Eigen::RowVectorXd calc = Geometry::GetElements(m_dataSet[0][0].row(0), idx); // row = -3, -4, -5, -4, -6, -1, -4, -1, -3, -1
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("GetElements", ref, calc);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Basics, ARange)
{
	const std::vector<size_t> ref{ 1, 3, 5, 7, 9 },
							  calc = Geometry::ARange(size_t(1), size_t(10), size_t(2));
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("ARange", ref, calc);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Basics, Vector2DTo1D)
{
	std::vector<Eigen::MatrixXd> calc = Geometry::Vector2DTo1D(m_dataSet);
	bool equal                        = true;
	size_t idx                        = 0;
	for (auto& set : m_dataSet) { for (const auto& data : set) { if (!isAlmostEqual(data, calc[idx++])) { equal = false; } } }
	EXPECT_TRUE(equal) << "Vector2DTo1D fail";
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Basics, Metrics)
{
	EXPECT_TRUE(toString(Geometry::EMetric::Riemann) == "Riemann");
	EXPECT_TRUE(toString(Geometry::EMetric::Euclidian) == "Euclidian");
	EXPECT_TRUE(toString(Geometry::EMetric::LogEuclidian) == "Log Euclidian");
	EXPECT_TRUE(toString(Geometry::EMetric::LogDet) == "Log Determinant");
	EXPECT_TRUE(toString(Geometry::EMetric::Kullback) == "Kullback");
	EXPECT_TRUE(toString(Geometry::EMetric::ALE) == "AJD-based log-Euclidean");
	EXPECT_TRUE(toString(Geometry::EMetric::Harmonic) == "Harmonic");
	EXPECT_TRUE(toString(Geometry::EMetric::Wasserstein) == "Wasserstein");
	EXPECT_TRUE(toString(Geometry::EMetric::Identity) == "Identity");
	EXPECT_TRUE(Geometry::StringToMetric("Riemann") == Geometry::EMetric::Riemann);
	EXPECT_TRUE(Geometry::StringToMetric("Euclidian") == Geometry::EMetric::Euclidian);
	EXPECT_TRUE(Geometry::StringToMetric("Log Euclidian") == Geometry::EMetric::LogEuclidian);
	EXPECT_TRUE(Geometry::StringToMetric("Log Determinant") == Geometry::EMetric::LogDet);
	EXPECT_TRUE(Geometry::StringToMetric("Kullback") == Geometry::EMetric::Kullback);
	EXPECT_TRUE(Geometry::StringToMetric("AJD-based log-Euclidean") == Geometry::EMetric::ALE);
	EXPECT_TRUE(Geometry::StringToMetric("Harmonic") == Geometry::EMetric::Harmonic);
	EXPECT_TRUE(Geometry::StringToMetric("Wasserstein") == Geometry::EMetric::Wasserstein);
	EXPECT_TRUE(Geometry::StringToMetric("Identity") == Geometry::EMetric::Identity);
	EXPECT_TRUE(Geometry::StringToMetric("") == Geometry::EMetric::Identity);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Basics, Validation)
{
	const Eigen::MatrixXd m1 = Eigen::MatrixXd::Zero(2, 2),
						  m2 = Eigen::MatrixXd::Zero(1, 2),
						  m3;
	std::vector<Eigen::MatrixXd> v;

	EXPECT_TRUE(Geometry::InRange(1, 0, 2));		// 0 <= 1 <= 2 ?
	EXPECT_FALSE(Geometry::InRange(2, 0, 1));		// 0 <= 2 <= 1 ?

	EXPECT_FALSE(Geometry::AreNotEmpty(v));	// Empty Vector
	v.push_back(m3);
	EXPECT_FALSE(Geometry::AreNotEmpty(v));	// Vector with one empty matix
	v.push_back(m1);
	EXPECT_FALSE(Geometry::AreNotEmpty(v));	// Vector With one empty matrix and one non empty matrix
	v.clear();
	v.push_back(m1);
	v.push_back(m2);
	EXPECT_TRUE(Geometry::AreNotEmpty(v));	// Vector With two non empty matrix

	EXPECT_TRUE(Geometry::HaveSameSize(m1, m1));	// Same matrix
	EXPECT_FALSE(Geometry::HaveSameSize(m3, m3));	// Same but empty
	EXPECT_FALSE(Geometry::HaveSameSize(m1, m2));	// DIfferents
	EXPECT_FALSE(Geometry::HaveSameSize(m1, m3));	// One empty

	EXPECT_FALSE(Geometry::HaveSameSize(v));		// Two different
	EXPECT_FALSE(Geometry::AreSquare(v));			// One square
	v.clear();
	v.push_back(m1);
	v.push_back(m1);
	EXPECT_TRUE(Geometry::HaveSameSize(v) && Geometry::AreSquare(v));	// Same matrix

	Geometry::MatrixPrint(m1);	// Only to check
	Geometry::MatrixPrint(m3);	// Only to check

	std::vector<std::string> vs = Geometry::Split("0,1,2,3.a\n", ",");
	EXPECT_TRUE(vs.size() == 4 && vs[0] == "0" && vs[1] == "1" && vs[2] == "2" && vs[3] == "3.a") << vs.size() << " " << vs[0] << " " << vs[1] << " " << vs[2] << " " << vs[3] << std::endl;
}
//---------------------------------------------------------------------------------------------------
