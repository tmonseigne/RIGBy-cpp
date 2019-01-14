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

//---------------------------------------------------------------------------------------------------
class Tests_Basics : public testing::Test
{
protected:
	std::vector<std::vector<Eigen::MatrixXd>> m_dataSet;

	void SetUp() override { m_dataSet = InitDataset::Dataset(); }
};
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Basics, MatrixCenter)
{
	std::vector<std::vector<Eigen::MatrixXd>> calc, ref = InitBasics::Center::Reference();
	calc.resize(m_dataSet.size());
	for (size_t k = 0; k < m_dataSet.size(); ++k)
	{
		calc[k].resize(m_dataSet[k].size());
		for (size_t i = 0; i < m_dataSet[k].size(); ++i)
		{
			MatrixCenter(m_dataSet[k][i], calc[k][i]);
			const std::string title = "Matrix Center Sample [" + std::to_string(k) + "][" + std::to_string(i) + "]";
			EXPECT_TRUE(isAlmostEqual(ref[k][i], calc[k][i])) << ErrorMsg(title, ref[k][i], calc[k][i]).str();
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
