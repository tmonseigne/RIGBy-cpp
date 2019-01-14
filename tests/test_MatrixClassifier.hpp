///-------------------------------------------------------------------------------------------------
/// 
/// \file test_MatrixClassifier.hpp
/// \brief Tests for Riemannian Geometry Matrix Classifier
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 09/01/2019.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// \remarks 
/// - For this tests I compare the results with the <a href="https://github.com/alexandrebarachant/pyRiemann">pyRiemann</a> library (<a href="https://github.com/alexandrebarachant/pyRiemann/blob/master/LICENSE">License</a>) or <a href="http://scikit-learn.org">sklearn</a> if pyRiemman just redirect the function.
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include "gtest/gtest.h"
#include "test_Misc.hpp"
#include "test_Init.hpp"
#include "classifier/CMatrixClassifierMDM.hpp"

//---------------------------------------------------------------------------------------------------
class Tests_MatrixClassifier : public testing::Test
{
protected:
	std::vector<std::vector<Eigen::MatrixXd>> m_dataSet;

	void SetUp() override { m_dataSet = InitCovariance::LWF::Reference(); }
};
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_MatrixClassifier, MDMTrain)
{
	const CMatrixClassifierMDM ref = InitMatrixClassif::MDM::Reference();
	CMatrixClassifierMDM calc;
	EXPECT_TRUE(calc.train(m_dataSet)) << "Error during Training : " << std::endl << calc << std::endl;
	EXPECT_TRUE(ref == calc) << ErrorMsg("MDMTrain", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
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
			EXPECT_TRUE(refPrediction[idx] == classid) << ErrorMsg("Prediction " + text, refPrediction[idx], classid).str();
			EXPECT_TRUE(isAlmostEqual(refPredictionDistance[idx],distance)) << ErrorMsg("Prediction Distance " + text, refPredictionDistance[idx], distance).str();
			idx++;
			result(k, classid)++;
		}
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_MatrixClassifier, MDMSave)
{
	CMatrixClassifierMDM calc, ref;
	EXPECT_TRUE(ref.train(m_dataSet)) << "Error during Training : " << std::endl << ref << std::endl;
	EXPECT_TRUE(ref.saveXML("test.xml")) << "Error during Saving : " << std::endl << ref << std::endl;
	EXPECT_TRUE(calc.loadXML("test.xml")) << "Error during Loading : " << std::endl << calc << std::endl;
	EXPECT_TRUE(ref == calc) << ErrorMsg("MDMSave", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_MatrixClassifier, FgMDMTrain)
{
	const CMatrixClassifierFgMDM ref = InitMatrixClassif::FgMDM::Reference();
	CMatrixClassifierFgMDM calc;
	EXPECT_TRUE(calc.train(m_dataSet)) << "Error during Training : " << std::endl << calc << std::endl;
	EXPECT_TRUE(ref == calc) << ErrorMsg("FgMDMTrain", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_MatrixClassifier, FgMDMClassifify)
{
	CMatrixClassifierFgMDM calc;
	EXPECT_TRUE(calc.train(m_dataSet)) << "Error during Training : " << std::endl << calc << std::endl;

	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(NB_CLASS, NB_CLASS);
	std::vector<size_t> refPrediction = InitMatrixClassif::FgMDM::Prediction();
	std::vector<std::vector<double>> refPredictionDistance = InitMatrixClassif::FgMDM::PredictionDistance();

	size_t idx = 0;
	for (size_t k = 0; k < m_dataSet.size(); ++k)
	{
		for (size_t i = 0; i < m_dataSet[k].size(); ++i)
		{
			size_t classid;
			std::vector<double> distance, probability;
			EXPECT_TRUE(calc.classify(m_dataSet[k][i], classid, distance, probability)) << "Error during Classify : " << std::endl << calc << std::endl;

			const std::string text = "sample [" + std::to_string(k) + "][" + std::to_string(i) + "] different";
			EXPECT_TRUE(refPrediction[idx] == classid) << ErrorMsg("Prediction " + text, refPrediction[idx], classid).str();
			EXPECT_TRUE(isAlmostEqual(refPredictionDistance[idx], distance)) << ErrorMsg("Prediction Distance " + text, refPredictionDistance[idx], distance).str();
			idx++;
			result(k, classid)++;
		}
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_MatrixClassifier, FgMDMSave)
{
	CMatrixClassifierFgMDM calc, ref;
	EXPECT_TRUE(ref.train(m_dataSet)) << "Error during Training : " << std::endl << ref << std::endl;
	EXPECT_TRUE(ref.saveXML("test.xml")) << "Error during Saving : " << std::endl << ref << std::endl;
	EXPECT_TRUE(calc.loadXML("test.xml")) << "Error during Loading : " << std::endl << calc << std::endl;
	EXPECT_TRUE(ref == calc) << ErrorMsg("FgMDMSave", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------
