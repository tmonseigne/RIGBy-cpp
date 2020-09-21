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
#include "test_Init.hpp"
#include "test_Misc.hpp"

#include "artifacts/CASR.hpp"
#include "utils/Basics.hpp"

//---------------------------------------------------------------------------------------------------
class Tests_ASR : public testing::Test
{
protected:
	std::vector<Eigen::MatrixXd> m_dataset;

	void SetUp() override { m_dataset = Vector2DTo1D(InitDataset::Dataset()); }
};
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_ASR, trainASR)
{
	CASR asr;
	asr.train(m_dataset);
	std::cout << asr << std::endl;

	Eigen::MatrixXd refMedian(3, 3), refTransformation(3, 3);
	refMedian << 1.32267188, 0.00105802, 0.00871490,
			0.00105802, 1.32447262, 0.01829110,
			0.00871490, 0.01829110, 1.02823244;
	refTransformation << -0.05738397, -0.12038745, 1.96253197,
			3.44884211, -1.60796319, 0.00220615,
			1.31682195, 2.82467797, 0.21177757;

	EXPECT_TRUE(asr.getMetric()== EMetric::Euclidian) << "Asr Train Metric : Reference : " << toString(EMetric::Euclidian)
			<< ", \tCompute : " << toString(asr.getMetric());
	EXPECT_TRUE(isAlmostEqual(asr.getMedian(), refMedian)) << ErrorMsg("Asr Train Median", asr.getMedian(), refMedian);
	EXPECT_TRUE(isAlmostEqual(asr.getTransformMatrix(), refTransformation))
			<< ErrorMsg("Asr Train Transformation Matrix", asr.getTransformMatrix(), refTransformation);
}
//---------------------------------------------------------------------------------------------------
