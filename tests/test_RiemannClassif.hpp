#pragma once

#include "gtest/gtest.h"
#include "test_Misc.hpp"
#include <random>
#include "classifier/CMatrixClassifierMDM.hpp"
#include "classifier/CMatrixClassifierFgMDM.hpp"


using namespace Eigen;
using namespace std;

#define NB_CLASS	4
#define NB_TRIALS	20
#define NB_CHAN		16
#define NB_SAMPLE	100

class Tests_Classifier : public testing::Test
{
protected:
	vector<vector<MatrixXd>> m_dataSet;

	static void initNormalDistribution(MatrixXd& m, double mu, double sigma)
	{
		m.resize(NB_CHAN, NB_SAMPLE);
		std::default_random_engine generator;
		normal_distribution<double> distribution(mu, sigma);

		for (size_t i = 0, size = m.size(); i < size; i++)
		{
			(*(m.data() + i)) = distribution(generator);
		}
	}

	static void initSet(vector<MatrixXd>& set, const double mu, const double sigma)
	{
		set.resize(NB_TRIALS);
		for (auto& s : set)
		{
			MatrixXd sample;
			initNormalDistribution(sample, mu, sigma);
			CovarianceMatrix(sample, s, Estimator_LWF);
		}
	}

	void SetUp() override
	{
		vector<double> mus = { 1, 2.2, 1.5, 0.9 };
		vector<double> sigmas = { 0.75, 0.8, 0.6, 0.7 };
		m_dataSet.resize(NB_CLASS);
		for (size_t i = 0; i < NB_CLASS; ++i)
		{
			initSet(m_dataSet[i], mus[i], sigmas[i]);
		}
	}
};

TEST_F(Tests_Classifier, TrainMDM)
{
	CMatrixClassifierMDM calc;
	EXPECT_TRUE(calc.train(m_dataSet)) << "Error during Training : " << endl << calc << endl;
	EXPECT_TRUE(calc.saveXML("test.xml"));
	//cout << "Classifier : " << endl << calc << endl;
}

TEST_F(Tests_Classifier, ClassififyMDM)
{
	CMatrixClassifierMDM calc;
	EXPECT_TRUE(calc.train(m_dataSet)) << "Error during Training : " << endl << calc << endl;
	MatrixXd result = MatrixXd::Zero(NB_CLASS, NB_CLASS);

	for (size_t i = 0; i < NB_CLASS; ++i)
	{
		for (size_t j = 0; j < NB_TRIALS; ++j)
		{
			size_t classid;
			EXPECT_TRUE(calc.classify(m_dataSet[i][j], classid)) << "Error during Classify : " << endl << calc << endl;
			result(i, classid)++;
		}
	}
	cout << "Result : " << endl << result << endl;
}

TEST_F(Tests_Classifier, XMLMDM)
{
	CMatrixClassifierMDM calc, ref;
	EXPECT_TRUE(ref.train(m_dataSet)) << "Error during Training : " << endl << ref << endl;
	EXPECT_TRUE(ref.saveXML("test.xml"));
	EXPECT_TRUE(calc.loadXML("test.xml"));
	EXPECT_TRUE(calc.saveXML("test2.xml"));
	EXPECT_TRUE(ref == calc);
}

TEST_F(Tests_Classifier, TrainFgMDM)
{
	CMatrixClassifierFgMDM calc;
	EXPECT_TRUE(calc.train(m_dataSet)) << "Error during Training : " << endl << calc << endl;
	EXPECT_TRUE(calc.saveXML("test.xml"));
	//cout << "Classifier : " << endl << calc << endl;
}
