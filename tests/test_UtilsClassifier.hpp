#pragma once

#include "gtest/gtest.h"
#include "test_Misc.hpp"
#include "test_Init.hpp"
#include "utils/Classification.hpp"

class Tests_Classifier : public testing::Test
{
protected:
	std::vector<std::vector<Eigen::RowVectorXd>> m_dataSet;

	void SetUp() override
	{
		std::vector<Eigen::RowVectorXd> tmp = InitFeaturization::TangentSpace::Dataset();
		m_dataSet = Vector1DTo2D(tmp, { NB_TRIALS1, NB_TRIALS2 });
	}
};

TEST_F(Tests_Classifier, LSQR)
{
	Eigen::MatrixXd calc, ref = InitClassif::LSQR::Dataset();
	LSQR(m_dataSet, calc);
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("LSQR", ref, calc).str();
}

TEST_F(Tests_Classifier, FgDACompute)
{
	Eigen::MatrixXd calc, ref = InitClassif::FgDACompute::Dataset();
	FgDACompute(m_dataSet, calc);
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("LSQR", ref, calc).str();
}
