///-------------------------------------------------------------------------------------------------
/// 
/// \file test_ASR.hpp
/// \brief Tests for Artifact Subspace Reconstruction.
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 29/07/2020.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// \remarks We use the EEglab Matlab plugin to compare result for validation
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include "gtest/gtest.h"
#include "Init.hpp"
#include "misc.hpp"

#include <geometry/artifacts/CASR.hpp>
#include <geometry/Basics.hpp>

//---------------------------------------------------------------------------------------------------
class Tests_ASR : public testing::Test
{
protected:
	std::vector<Eigen::MatrixXd> m_dataset;

	void SetUp() override { m_dataset = Geometry::Vector2DTo1D(InitDataset::Dataset()); }
};

//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_ASR, ASR_Train_Euclidian)
{
	const Geometry::CASR ref = InitASR::Euclidian::Reference();
	const Geometry::CASR calc(Geometry::EMetric::Euclidian, m_dataset);
	EXPECT_TRUE(calc == ref) << ErrorMsg("Train ASR in Euclidian metric", ref, calc);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_ASR, ASR_Train_Riemann)
{
	std::cout << "Riemannian Eigen Value isn't implemented, so result is same as Euclidian metric." << std::endl;
	const Geometry::CASR ref = InitASR::Riemann::Reference();
	const Geometry::CASR calc(Geometry::EMetric::Riemann, m_dataset);
	EXPECT_TRUE(calc == ref) << ErrorMsg("Train ASR in Riemann metric", ref, calc);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_ASR, ASR_Process)
{
	m_dataset = InitDataset::FirstClassDataset();
	Geometry::CASR calc(Geometry::EMetric::Euclidian, m_dataset);

	std::vector<Eigen::MatrixXd> testset = InitDataset::SecondClassDataset();
	std::vector<Eigen::MatrixXd> result(testset.size());
	for (size_t i = 0; i < testset.size(); ++i)
	{
		testset[i] *= 2;
		EXPECT_TRUE(calc.process(testset[i], result[i])) << "ASR PRocess fail for sample " + std::to_string(i) + ".\n";
	}
	for (size_t i = 1; i < testset.size(); ++i)
	{
		EXPECT_FALSE(isAlmostEqual(result[i], testset[i])) << "the sample " + std::to_string(i) + " wasn't reconstructed.\n";
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_ASR, ASR_Save)
{
	//Geometry::CASR calc;
	//const Geometry::CASR ref = InitASR::Euclidian::Reference();
	//EXPECT_TRUE(ref.saveXML("test_ASR_Save.xml")) << "Error during Saving : " << std::endl << ref << std::endl;
	//EXPECT_TRUE(calc.loadXML("test_ASR_Save.xml")) << "Error during Loading : " << std::endl << calc << std::endl;
	//EXPECT_TRUE(ref == calc) << ErrorMsg("ASR Save", ref, calc);
}
//---------------------------------------------------------------------------------------------------
