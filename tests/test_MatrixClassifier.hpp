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
/// - For the adaptation Classification tests I compare the results with the <a href="https://github.com/alexandrebarachant/covariancetoolbox">covariancetoolbox</a> Matlab library (<a href="https://github.com/alexandrebarachant/covariancetoolbox/blob/master/COPYING">License</a>).
/// - The Matlab toolbox is older and Riemannian mean estimation is différent the test are adapted to switch between the two library
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include "gtest/gtest.h"
#include "test_Misc.hpp"
#include "test_Init.hpp"
#include "classifier/CMatrixClassifierMDM.hpp"
#include "classifier/CMatrixClassifierMDMRebias.hpp"
#include "classifier/CMatrixClassifierFgMDM.hpp"

//---------------------------------------------------------------------------------------------------
static void TestClassify(IMatrixClassifier& calc, const std::vector<std::vector<Eigen::MatrixXd>>& dataset, const std::vector<size_t>& refPrediction, const std::vector<std::vector<double>>& refPredictionDistance, const EAdaptations& adapt)
{
	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(NB_CLASS, NB_CLASS);
	size_t idx             = 0;
	for (size_t k = 0; k < dataset.size(); ++k)
	{
		for (size_t i = 0; i < dataset[k].size(); ++i)
		{
			const std::string text = "sample [" + std::to_string(k) + "][" + std::to_string(i) + "]";
			size_t classid         = 0;
			std::vector<double> distance, probability;
			EXPECT_TRUE(calc.classify(dataset[k][i], classid, distance, probability, adapt, k)) << "Error during Classify " << text;
			if (idx < refPrediction.size())
			{
				EXPECT_TRUE(refPrediction[idx] == classid) << ErrorMsg("Prediction " + text, refPrediction[idx], classid).str();
			}
			if (idx < refPredictionDistance.size())
			{
				EXPECT_TRUE(isAlmostEqual(refPredictionDistance[idx], distance)) << ErrorMsg("Prediction Distance " + text, refPredictionDistance[idx], distance).str();
			}
			idx++;
			result(k, classid)++;
		}
	}
	std::cout << "***** Classifier : *****" << std::endl << calc << std::endl << "***** Result : *****" << std::endl << result << std::endl;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
class Tests_MatrixClassifier : public testing::Test
{
protected:
	std::vector<std::vector<Eigen::MatrixXd>> m_dataSet;

	void SetUp() override { m_dataSet = InitCovariance::LWF::Reference(); }
};
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_MatrixClassifier, MDM_Train)
{
	const CMatrixClassifierMDM ref = InitMatrixClassif::MDM::Reference();
	CMatrixClassifierMDM calc;
	EXPECT_TRUE(calc.train(m_dataSet)) << "Error during Training : " << std::endl << calc << std::endl;
	EXPECT_TRUE(ref == calc) << ErrorMsg("MDM Train", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_MatrixClassifier, MDM_Classifify)
{
	CMatrixClassifierMDM calc = InitMatrixClassif::MDM::ReferenceMatlab();
	TestClassify(calc, m_dataSet, InitMatrixClassif::MDM::Prediction(), InitMatrixClassif::MDM::PredictionDistance(), Adaptation_None);
	const CMatrixClassifierMDM ref = InitMatrixClassif::MDM::ReferenceMatlab();	// No Change
	EXPECT_TRUE(ref == calc) << ErrorMsg("MDM Classify Change without adaptation mode", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_MatrixClassifier, MDM_Classifify_Adapt_Supervised)
{
	CMatrixClassifierMDM calc = InitMatrixClassif::MDM::ReferenceMatlab();
	TestClassify(calc, m_dataSet, InitMatrixClassif::MDM::PredictionSupervised(), InitMatrixClassif::MDM::PredictionDistanceSupervised(), Adaptation_Supervised);
	const CMatrixClassifierMDM ref = InitMatrixClassif::MDM::AfterSupervised();
	EXPECT_TRUE(ref == calc) << ErrorMsg("MDM Adapt Classify after Supervised adaptation", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_MatrixClassifier, MDM_Classifify_Adapt_Unsupervised)
{
	CMatrixClassifierMDM calc = InitMatrixClassif::MDM::ReferenceMatlab();
	TestClassify(calc, m_dataSet, InitMatrixClassif::MDM::PredictionUnSupervised(), InitMatrixClassif::MDM::PredictionDistanceUnSupervised(), Adaptation_Unsupervised);
	const CMatrixClassifierMDM ref = InitMatrixClassif::MDM::AfterUnSupervised();
	EXPECT_TRUE(ref == calc) << ErrorMsg("MDM Adapt Classify after Unsupervised adaptation", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_MatrixClassifier, MDM_Save)
{
	CMatrixClassifierMDM calc;
	CMatrixClassifierMDM ref = InitMatrixClassif::MDM::Reference();
	EXPECT_TRUE(ref.saveXML("test_MDM_Save.xml")) << "Error during Saving : " << std::endl << ref << std::endl;
	EXPECT_TRUE(calc.loadXML("test_MDM_Save.xml")) << "Error during Loading : " << std::endl << calc << std::endl;
	EXPECT_TRUE(ref == calc) << ErrorMsg("MDM Save", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_MatrixClassifier, FgMDMRT_Train)
{
	const CMatrixClassifierFgMDMRT ref = InitMatrixClassif::FgMDMRT::Reference();
	CMatrixClassifierFgMDMRT calc;
	EXPECT_TRUE(calc.train(m_dataSet)) << "Error during Training : " << std::endl << calc << std::endl;
	EXPECT_TRUE(ref == calc) << ErrorMsg("FgMDM Train", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_MatrixClassifier, FgMDMRT_Classifify)
{
	CMatrixClassifierFgMDMRT calc = InitMatrixClassif::FgMDMRT::Reference();
	TestClassify(calc, m_dataSet, InitMatrixClassif::FgMDMRT::Prediction(), InitMatrixClassif::FgMDMRT::PredictionDistance(), Adaptation_None);
	const CMatrixClassifierFgMDMRT ref = InitMatrixClassif::FgMDMRT::Reference();
	EXPECT_TRUE(ref == calc) << ErrorMsg("FgMDM Classify Change without adaptation mode", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_MatrixClassifier, FgMDMRT_Classifify_Adapt_Supervised)
{
	CMatrixClassifierFgMDMRT calc = InitMatrixClassif::FgMDMRT::Reference();
	const std::vector<std::vector<double>> emptydist;
	TestClassify(calc, m_dataSet, InitMatrixClassif::FgMDMRT::PredictionSupervised(), emptydist, Adaptation_Supervised);
	//const CMatrixClassifierFgMDMRT ref = InitMatrixClassif::FgMDMRT::AfterSupervised();
	//EXPECT_TRUE(ref == calc) << ErrorMsg("FgMDM Adapt Classify after Supervised RT adaptation", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_MatrixClassifier, FgMDMRT_Classifify_Adapt_Unsupervised)
{
	CMatrixClassifierFgMDMRT calc(InitMatrixClassif::FgMDMRT::Reference());
	const std::vector<std::vector<double>> emptydist;
	TestClassify(calc, m_dataSet, InitMatrixClassif::FgMDMRT::PredictionUnSupervised(), emptydist, Adaptation_Unsupervised);
	//const CMatrixClassifierFgMDMRT ref = InitMatrixClassif::FgMDMRT::AfterUnSupervised();
	//EXPECT_TRUE(ref == calc) << ErrorMsg("FgMDM Adapt Classify after Unsupervised RT adaptation", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_MatrixClassifier, FgMDMRT_Save)
{
	CMatrixClassifierFgMDMRT calc;
	CMatrixClassifierFgMDMRT ref = InitMatrixClassif::FgMDMRT::Reference();
	EXPECT_TRUE(ref.saveXML("test_FgMDM_Save.xml")) << "Error during Saving : " << std::endl << ref << std::endl;
	EXPECT_TRUE(calc.loadXML("test_FgMDM_Save.xml")) << "Error during Loading : " << std::endl << calc << std::endl;
	EXPECT_TRUE(ref == calc) << ErrorMsg("FgMDM Save", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_MatrixClassifier, FgMDM_Classifify_Adapt_Supervised)
{
	CMatrixClassifierFgMDM calc = InitMatrixClassif::FgMDM::Reference();
	const std::vector<std::vector<double>> emptydist;
	TestClassify(calc, m_dataSet, InitMatrixClassif::FgMDM::PredictionSupervised(), emptydist, Adaptation_Supervised);
	//const CMatrixClassifierFgMDM ref = InitMatrixClassif::FgMDM::AfterSupervised();
	//EXPECT_TRUE(ref == calc) << ErrorMsg("FgMDM Adapt Classify after Supervised adaptation", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_MatrixClassifier, FgMDM_Classifify_Adapt_Unsupervised)
{
	CMatrixClassifierFgMDM calc = InitMatrixClassif::FgMDM::Reference();
	const std::vector<std::vector<double>> emptydist;
	TestClassify(calc, m_dataSet, InitMatrixClassif::FgMDM::PredictionUnSupervised(), emptydist, Adaptation_Unsupervised);
	//const CMatrixClassifierFgMDM ref = InitMatrixClassif::FgMDM::AfterUnSupervised();
	//EXPECT_TRUE(ref == calc) << ErrorMsg("FgMDM Adapt Classify after Unsupervised adaptation", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_MatrixClassifier, MDM_Rebias_Train)
{
	CMatrixClassifierMDMRebias calc;
	EXPECT_TRUE(calc.train(m_dataSet)) << "Error during Training : " << std::endl << calc << std::endl;
	//const CMatrixClassifierMDMRebias ref = InitMatrixClassif::MDMRebias::Reference();
	//EXPECT_TRUE(ref == calc) << ErrorMsg("MDM Rebias Train", ref, calc).str(); // The mean method is different in matlab toolbox and python toolbox
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_MatrixClassifier, MDM_Rebias_Classifify)
{
	CMatrixClassifierMDMRebias calc = InitMatrixClassif::MDMRebias::Reference();
	TestClassify(calc, m_dataSet, InitMatrixClassif::MDMRebias::Prediction(), InitMatrixClassif::MDMRebias::PredictionDistance(), Adaptation_None);
	const CMatrixClassifierMDMRebias ref = InitMatrixClassif::MDMRebias::After();	// No Class change but Rebias yes
	EXPECT_TRUE(ref == calc) << ErrorMsg("MDM Rebias Classify Change without adaptation mode", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_MatrixClassifier, MDM_Rebias_Classifify_Adapt_Supervised)
{
	CMatrixClassifierMDMRebias calc = InitMatrixClassif::MDMRebias::Reference();
	TestClassify(calc, m_dataSet, InitMatrixClassif::MDMRebias::PredictionSupervised(), InitMatrixClassif::MDMRebias::PredictionDistanceSupervised(), Adaptation_Supervised);
	const CMatrixClassifierMDMRebias ref = InitMatrixClassif::MDMRebias::AfterSupervised();
	EXPECT_TRUE(ref == calc) << ErrorMsg("MDM Rebias Adapt Classify after Supervised adaptation", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_MatrixClassifier, MDM_Rebias_Classifify_Adapt_Unsupervised)
{
	CMatrixClassifierMDMRebias calc = InitMatrixClassif::MDMRebias::Reference();
	TestClassify(calc, m_dataSet, InitMatrixClassif::MDMRebias::PredictionUnSupervised(), InitMatrixClassif::MDMRebias::PredictionDistanceUnSupervised(), Adaptation_Unsupervised);
	const CMatrixClassifierMDMRebias ref = InitMatrixClassif::MDMRebias::AfterUnSupervised();
	EXPECT_TRUE(ref == calc) << ErrorMsg("MDM Rebias Adapt Classify after Unsupervised adaptation", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------


//---------------------------------------------------------------------------------------------------
TEST_F(Tests_MatrixClassifier, MDM_Rebias_Save)
{
	CMatrixClassifierMDMRebias calc;
	CMatrixClassifierMDMRebias ref = InitMatrixClassif::MDMRebias::Reference();
	EXPECT_TRUE(ref.saveXML("test_MDM_Rebias_Save.xml")) << "Error during Saving : " << std::endl << ref << std::endl;
	EXPECT_TRUE(calc.loadXML("test_MDM_Rebias_Save.xml")) << "Error during Loading : " << std::endl << calc << std::endl;
	EXPECT_TRUE(ref == calc) << ErrorMsg("MDM Rebias Save", ref, calc).str();
}
//---------------------------------------------------------------------------------------------------
