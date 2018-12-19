#pragma once

#include "gtest/gtest.h"
#include "test_Misc.hpp"
#include "test_Init.hpp"
#include "utils/Covariance.hpp"

//---------------------------------------------------------------------------------------------------
class Tests_Covariances : public testing::Test
{
protected:
	std::vector<std::vector<Eigen::MatrixXd>> m_dataSet;

	void SetUp() override { m_dataSet = InitDataset::Dataset(); }
};
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Covariances, Covariance_Matrix_COR)
{
	std::vector<std::vector<Eigen::MatrixXd>> calc, ref = InitCovariance::COR::Dataset();
	calc.resize(m_dataSet.size());
	for (size_t k = 0; k < m_dataSet.size(); ++k)
	{
		calc[k].resize(m_dataSet[k].size());
		for (size_t i = 0; i < m_dataSet[k].size(); ++i)
		{
			CovarianceMatrix(m_dataSet[k][i], calc[k][i], Estimator_COR, Standardization_None);
			const std::string title = "Covariance Matrix COR Sample [" + std::to_string(k) + "][" + std::to_string(i) + "]";
			EXPECT_TRUE(isAlmostEqual(ref[k][i], calc[k][i])) << ErrorMsg(title, ref[k][i], calc[k][i]).str();
		}
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Covariances, Covariance_Matrix_COV)
{
	std::vector<std::vector<Eigen::MatrixXd>> calc, ref = InitCovariance::COV::Dataset();
	calc.resize(m_dataSet.size());
	for (size_t k = 0; k < m_dataSet.size(); ++k)
	{
		calc[k].resize(m_dataSet[k].size());
		for (size_t i = 0; i < m_dataSet[k].size(); ++i)
		{
			CovarianceMatrix(m_dataSet[k][i], calc[k][i], Estimator_COV, Standardization_None);
			const std::string title = "Covariance Matrix COV Sample [" + std::to_string(k) + "][" + std::to_string(i) + "]";
			EXPECT_TRUE(isAlmostEqual(ref[k][i], calc[k][i])) << ErrorMsg(title, ref[k][i], calc[k][i]).str();
		}
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Covariances, Covariance_Matrix_LWF)
{
	std::vector<std::vector<Eigen::MatrixXd>> calc, ref = InitCovariance::LWF::Dataset();
	calc.resize(m_dataSet.size());
	for (size_t k = 0; k < m_dataSet.size(); ++k)
	{
		calc[k].resize(m_dataSet[k].size());
		for (size_t i = 0; i < m_dataSet[k].size(); ++i)
		{
			CovarianceMatrix(m_dataSet[k][i], calc[k][i], Estimator_LWF, Standardization_Center);
			const std::string title = "Covariance Matrix LWF Sample [" + std::to_string(k) + "][" + std::to_string(i) + "]";
			EXPECT_TRUE(isAlmostEqual(ref[k][i], calc[k][i])) << ErrorMsg(title, ref[k][i], calc[k][i]).str();
		}
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Covariances, Covariance_Matrix_MCD)
{
	EXPECT_TRUE(false) << "Not implemented";
	/*
	vector<vector<MatrixXd>> calc, ref = InitCovariance::MCD::Dataset();
	calc.resize(m_dataSet.size());
	for (size_t k = 0; k < m_dataSet.size(); ++k)
	{
		calc[k].resize(m_dataSet[k].size());
		for (size_t i = 0; i < m_dataSet[k].size(); ++i)
		{
			CovarianceMatrix(m_dataSet[k][i], calc[k][i], Estimator_MCD, Standardization_Center);
			const std::string title = "Covariance Matrix MCD Sample [" + std::to_string(k) + "][" + std::to_string(i) + "]";
			EXPECT_TRUE(isAlmostEqual(ref[k][i], calc[k][i])) << ErrorMsg(title, ref[k][i], calc[k][i]).str();
		}
	}
	*/
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Covariances, Covariance_Matrix_OAS)
{
	std::vector<std::vector<Eigen::MatrixXd>> calc, ref = InitCovariance::OAS::Dataset();
	calc.resize(m_dataSet.size());
	for (size_t k = 0; k < m_dataSet.size(); ++k)
	{
		calc[k].resize(m_dataSet[k].size());
		for (size_t i = 0; i < m_dataSet[k].size(); ++i)
		{
			CovarianceMatrix(m_dataSet[k][i], calc[k][i], Estimator_OAS, Standardization_Center);
			const std::string title = "Covariance Matrix OAS Sample [" + std::to_string(k) + "][" + std::to_string(i) + "]";
			EXPECT_TRUE(isAlmostEqual(ref[k][i], calc[k][i])) << ErrorMsg(title, ref[k][i], calc[k][i]).str();
		}
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Covariances, Covariance_Matrix_SCM)
{
	std::vector<std::vector<Eigen::MatrixXd>> calc, ref = InitCovariance::SCM::Dataset();
	calc.resize(m_dataSet.size());
	for (size_t k = 0; k < m_dataSet.size(); ++k)
	{
		calc[k].resize(m_dataSet[k].size());
		for (size_t i = 0; i < m_dataSet[k].size(); ++i)
		{
			CovarianceMatrix(m_dataSet[k][i], calc[k][i], Estimator_SCM, Standardization_None);
			const std::string title = "Covariance Matrix SCM Sample [" + std::to_string(k) + "][" + std::to_string(i) + "]";
			EXPECT_TRUE(isAlmostEqual(ref[k][i], calc[k][i])) << ErrorMsg(title, ref[k][i], calc[k][i]).str();
		}
	}
}
//---------------------------------------------------------------------------------------------------
