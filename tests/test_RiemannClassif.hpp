#pragma once

#include "gtest/gtest.h"
#include "test_Misc.hpp"
#include <random>
#include "classifier/CMatrixClassifierMDM.hpp"


using namespace Eigen;
using namespace std;

const int NB_CLASS = 4, NB_TRIALS = 20, NB_CHAN = 16, NB_SAMPLE = 100;
vector<MatrixXd> Set1(NB_TRIALS), Set2(NB_TRIALS);
vector<vector<MatrixXd>> DataSet(NB_CLASS);
vector<double> Mus(NB_CLASS), Sigmas(NB_CLASS);

inline void InitNormalDistribution(MatrixXd& m, double mu, double sigma)
{
	m.resize(NB_CHAN, NB_SAMPLE);
	std::default_random_engine generator;
	normal_distribution<double> distribution(mu, sigma);

	for (size_t i = 0, size = m.size(); i < size; i++)
	{
		(*(m.data() + i)) = distribution(generator);
	}
}

inline void InitSet(vector<MatrixXd>& set, const double mu, const double sigma)
{
	set.resize(NB_TRIALS);
	for (auto& s : set)
	{
		MatrixXd sample;
		InitNormalDistribution(sample, mu, sigma);
		CovarianceMatrix(sample, s, Estimator_LWF);
	}
}

inline void Init()
{
	Mus = { 1, 2.2, 1.5, 0.9 };
	Sigmas = { 0.75, 0.8, 0.6, 0.7 };
	for (size_t i = 0; i < NB_CLASS; ++i)
	{
		InitSet(DataSet[i], Mus[i], Sigmas[i]);
	}
}

class Classifier_Tests : public testing::Test {};

TEST_F(Classifier_Tests, TrainMDM)
{
	Init();
	CMatrixClassifierMDM calc;
	EXPECT_TRUE(calc.train(DataSet)) << "Error during Training : " << endl << calc << endl;
	EXPECT_TRUE(calc.saveXML("test.xml"));
	//cout << "Classifier : " << endl << calc << endl;
}


TEST_F(Classifier_Tests, ClassififyMDM)
{
	Init();
	CMatrixClassifierMDM calc;
	EXPECT_TRUE(calc.train(DataSet)) << "Error during Training : " << endl << calc << endl;
	MatrixXd result = MatrixXd::Zero(NB_CLASS, NB_CLASS);

	for (size_t i = 0; i < NB_CLASS; ++i)
	{
		for (size_t j = 0; j < NB_TRIALS; ++j)
		{
			uint32_t classid;
			EXPECT_TRUE(calc.classify(DataSet[i][j], classid)) << "Error during Classify : " << endl << calc << endl;
			result(i, classid)++;
		}
	}
	cout << "Result : " << endl << result << endl;
}

TEST_F(Classifier_Tests, XMLMDM)
{
	Init();
	CMatrixClassifierMDM calc, ref;
	EXPECT_TRUE(ref.train(DataSet)) << "Error during Training : " << endl << ref << endl;
	EXPECT_TRUE(ref.saveXML("test.xml"));
	EXPECT_TRUE(calc.loadXML("test.xml"));
	EXPECT_TRUE(calc.saveXML("test2.xml"));
	EXPECT_TRUE(ref == calc);
}