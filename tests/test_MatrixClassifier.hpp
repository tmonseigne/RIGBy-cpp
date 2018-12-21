#pragma once

#include "gtest/gtest.h"
#include "test_Misc.hpp"
#include "test_Init.hpp"
#include "classifier/CMatrixClassifierMDM.hpp"


class Tests_MatrixClassifier : public testing::Test
{
protected:
	std::vector<std::vector<Eigen::MatrixXd>> m_dataSet;

	void SetUp() override { m_dataSet = InitCovariance::LWF::Dataset(); }
};

TEST_F(Tests_MatrixClassifier, MDMTrain)
{
	std::vector<Eigen::MatrixXd> ref = InitMatrixClassif::MDM::Dataset();
	CMatrixClassifierMDM calc;
	EXPECT_TRUE(calc.train(m_dataSet)) << "Error during Training : " << std::endl << calc << std::endl;
	for (size_t k = 0; k < NB_CLASS; ++k)
	{
		const std::string title = "Mean Class " + std::to_string(k);
		EXPECT_TRUE(isAlmostEqual(ref[k], calc.m_Means[k])) << ErrorMsg(title, ref[k], calc.m_Means[k]).str();
	}
}

TEST_F(Tests_MatrixClassifier, MDMClassifify)
{
	CMatrixClassifierMDM calc;
	EXPECT_TRUE(calc.train(m_dataSet)) << "Error during Training : " << std::endl << calc << std::endl;
	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(NB_CLASS, NB_CLASS);
	std::vector<size_t> refPrediction = InitMatrixClassif::MDM::Prediction();
	std::vector<std::vector<double>> refPredictionDistance = InitMatrixClassif::MDM::PredictionDistance();
	size_t idx = 0;
	for (size_t k = 0; k < m_dataSet.size(); ++k)
	{
		for (size_t i = 0; i < m_dataSet[k].size(); ++i)
		{
			size_t classid;
			std::vector<double> distance, probability;
			EXPECT_TRUE(calc.classify(m_dataSet[k][i], classid, distance, probability)) << "Error during Classify : " << std::endl << calc << std::endl;
			const std::string text = "sample [" + std::to_string(k) + "][" + std::to_string(i) + "] different";
			EXPECT_TRUE(refPrediction[idx] == classid) << ErrorMsg("Prediction Distance " + text, refPrediction[idx], classid).str();
			EXPECT_TRUE(isAlmostEqual(refPredictionDistance[idx],distance)) << ErrorMsg("Prediction Distance " + text, refPredictionDistance[idx], distance).str();
			idx++;
			result(k, classid)++;
		}
	}
	std::cout << "Result : " << std::endl << result << std::endl;
}

TEST_F(Tests_MatrixClassifier, MDMSave)
{
	/*
	CMatrixClassifierMDM calc, ref;
	EXPECT_TRUE(ref.train(m_dataSet)) << "Error during Training : " << endl << ref << endl;
	EXPECT_TRUE(ref.saveXML("test.xml"));
	EXPECT_TRUE(calc.loadXML("test.xml"));
	EXPECT_TRUE(calc.saveXML("test2.xml"));
	EXPECT_TRUE(ref == calc);
	*/
}

/*
TEST_F(Tests_Classifier, TrainFgMDM)
{
	CMatrixClassifierFgMDM calc;
	m_dataSet.resize(2);
	m_dataSet[0].resize(4);
	EXPECT_TRUE(calc.train(m_dataSet)) << "Error during Training : " << endl << calc << endl;
	EXPECT_TRUE(calc.saveXML("test.xml"));
	//cout << "Classifier : " << endl << calc << endl;
}
*/
