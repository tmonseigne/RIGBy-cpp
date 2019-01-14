///-------------------------------------------------------------------------------------------------
/// 
/// \file test_UtilsClassifier.hpp
/// \brief Tests for Riemannian Geometry Utils : Classifier
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
#include "utils/Classification.hpp"
//---------------------------------------------------------------------------------------------------
class Tests_Classifier : public testing::Test
{
protected:
	std::vector<std::vector<Eigen::RowVectorXd>> m_dataSet;

	void SetUp() override
	{
		const std::vector<Eigen::RowVectorXd> tmp = InitFeaturization::TangentSpace::Reference();
		m_dataSet = Vector1DTo2D(tmp, { NB_TRIALS1, NB_TRIALS2 });
	}
};
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Classifier, LSQR)
{
	const Eigen::MatrixXd ref = InitClassif::LSQR::Reference();
	Eigen::MatrixXd calc;
	LSQR(m_dataSet, calc);
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("LSQR", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Classifier, FgDACompute)
{
	const Eigen::MatrixXd ref = InitClassif::FgDACompute::Reference();
	Eigen::MatrixXd calc;
	FgDACompute(m_dataSet, calc);
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("FgDA", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------
