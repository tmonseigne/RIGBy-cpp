#pragma once

#include "gtest/gtest.h"
#include "test_Misc.hpp"
#include "utils/Classification.hpp"

using namespace Eigen;
using namespace std;

class Tests_Classif : public testing::Test
{
protected:
	vector<vector<RowVectorXd>> m_dataset;

	void SetUp() override
	{
		m_dataset.resize(2);
		m_dataset[0].resize(4);
		m_dataset[1].resize(3);
		for (auto& i : m_dataset) { for (auto& j : i) { j.resize(2); } }

		m_dataset[0][0] << -1, -1;
		m_dataset[0][1] << -2, -1;
		m_dataset[0][2] << -3, -2;
		m_dataset[0][3] << -3, -1;

		m_dataset[1][0] << 1, 1;
		m_dataset[1][1] << 2, 1;
		m_dataset[1][2] << 3, 2;
	}
};

TEST_F(Tests_Classif, LSQR)
{
	MatrixXd calc, ref(2, 2);
	ref << -2.71806273, -5.11056363, 2.28145656, 5.69354645;
	LSQR(m_dataset, calc);
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("LSQR", ref, calc).str();
}
