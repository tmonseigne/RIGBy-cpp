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
TEST_F(Tests_ASR, trainASR)
{
	Geometry::CASR asr;
	asr.train(m_dataset);

	Eigen::MatrixXd refMedian(3, 3), refTransformation(3, 3);
	refMedian << 1.32267188, 0.00105802, 0.00871490,
			0.00105802, 1.32447262, 0.01829110,
			0.00871490, 0.01829110, 1.02823244;
	refTransformation << -0.05738723, - 0.12041171, 1.96255648,
			3.44737720, - 1.60950341, 0.00205466,
			1.31840128, 2.82413924, 0.21182516;

	EXPECT_TRUE(asr.getMetric() == Geometry::EMetric::Euclidian) << "Asr Train Metric : Reference : " << toString(Geometry::EMetric::Euclidian)
		<< ", \tCompute : " << toString(asr.getMetric());
	EXPECT_TRUE(isAlmostEqual(asr.getMedian(), refMedian)) << ErrorMsg("Asr Train Median", refMedian, asr.getMedian());
	EXPECT_TRUE(isAlmostEqual(asr.getTransformMatrix(), refTransformation))
		<< ErrorMsg("Asr Train Transformation Matrix", refTransformation, asr.getTransformMatrix());
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_ASR, ProcessASR)
{
	Geometry::CASR asr;
	m_dataset = InitDataset::FirstClassDataset();
	asr.train(m_dataset);

	std::vector<Eigen::MatrixXd> testset = InitDataset::SecondClassDataset();
	std::vector<Eigen::MatrixXd> result(testset.size());
	for (size_t i = 0; i < testset.size(); ++i)
	{
		testset[i] *= 2;
		EXPECT_TRUE(asr.process(testset[i], result[i])) << "ASR PRocess Fail\n";
	}
	for (size_t i = 1; i < testset.size(); ++i)
	{
		EXPECT_FALSE(isAlmostEqual(result[i], testset[i])) << "the sample " + std::to_string(i) + " wasn't reconstructed\n";
	}
}
//---------------------------------------------------------------------------------------------------
