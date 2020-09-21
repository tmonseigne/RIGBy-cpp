///-------------------------------------------------------------------------------------------------
/// 
/// \file test_UtilsMisc.hpp
/// \brief Tests for Utils : Misc
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

#include "utils/Misc.hpp"
#include "utils/Basics.hpp"

//---------------------------------------------------------------------------------------------------
class Tests_Misc : public testing::Test
{
	//protected:
	//	std::vector<Eigen::MatrixXd> m_dataSet;
	//
	//	void SetUp() override { m_dataSet = Vector2DTo1D(InitCovariance::LWF::Reference()); }
};
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Misc, SimpleMedian)
{
	std::vector<double> v{ 5, 6, 4, 3, 2, 6, 7, 9, 3 };
	double calc = Median(v);
	EXPECT_EQ(calc, 5);

	v.pop_back();
	calc = Median(v);
	EXPECT_EQ(calc, 5.5);

	Eigen::MatrixXd m(3, 3);
	m << 5, 6, 4, 3, 2, 6, 7, 9, 3;
	calc = Median(m);
	EXPECT_EQ(calc, 5);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Misc, DatasetMedian)
{
	const std::vector<Eigen::MatrixXd> dataSet = Vector2DTo1D(InitCovariance::LWF::Reference());
	Eigen::MatrixXd calc;
	Eigen::MatrixXd ref(3, 3);
	ref << 1.749537973777478, 0.002960131606861, 0.020507254841909,
			0.002960131606861, 1.754563395557952, 0.043042786354499,
			0.020507254841909, 0.043042786354499, 1.057672472691352;
	EXPECT_TRUE(Median(dataSet, calc)) << "Error During Median Computes";
	EXPECT_TRUE(isAlmostEqual(ref, calc)) << ErrorMsg("Median of Dataset", ref, calc);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Misc, DoubleRange)
{
	const std::vector<double> calc1 = doubleRange(0, 10, 2), calc2                 = doubleRange(0, 10, 2, false),
							  calc3 = doubleRange(0.15, 3.05, 0.5), calc4          = doubleRange(0.15, 3.05, 0.5, false),
							  ref1  = { 0, 2, 4, 6, 8, 10 }, ref2                  = { 0, 2, 4, 6, 8 },
							  ref3  = { 0.15, 0.65, 1.15, 1.65, 2.15, 2.65 }, ref4 = { 0.15, 0.65, 1.15, 1.65, 2.15, 2.65 };

	EXPECT_TRUE(isAlmostEqual(ref1, calc1)) << ErrorMsg("Double closed Range with integer value", ref1, calc1);
	EXPECT_TRUE(isAlmostEqual(ref2, calc2)) << ErrorMsg("Double opened Range with integer value", ref2, calc2);
	EXPECT_TRUE(isAlmostEqual(ref3, calc3)) << ErrorMsg("Double closed Range with double value", ref3, calc3);
	EXPECT_TRUE(isAlmostEqual(ref4, calc4)) << ErrorMsg("Double opened Range with double value", ref4, calc4);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Misc, RoundIndexRange)
{
	const std::vector<size_t> calc1 = RoundIndexRange(0, 10, 2), calc2        = RoundIndexRange(0, 10, 2, false),
							  calc3 = RoundIndexRange(0.15, 3.15, 0.2), calc4 = RoundIndexRange(0.15, 3.05, 0.2, false, false),
							  ref1  = { 0, 2, 4, 6, 8, 10 }, ref2             = { 0, 2, 4, 6, 8 },
							  ref3  = { 0, 1, 2, 3 }, ref4                    = { 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3 };

	EXPECT_TRUE(isAlmostEqual(ref1, calc1)) << ErrorMsg("Round Index closed Range with integer value", ref1, calc1);
	EXPECT_TRUE(isAlmostEqual(ref2, calc2)) << ErrorMsg("Round Index opened Range with integer value", ref2, calc2);
	EXPECT_TRUE(isAlmostEqual(ref3, calc3)) << ErrorMsg("Round Index closed Range with double value", ref3, calc3);
	EXPECT_TRUE(isAlmostEqual(ref4, calc4)) << ErrorMsg("Round Index opened Range with double value", ref4, calc4);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Misc, binHistogramm)
{
	//========== Create Dataset ==========
	const std::vector<Eigen::MatrixXd> matrices = Vector2DTo1D(InitDataset::Dataset());
	std::vector<std::vector<double>> dataset(NB_CHAN);
	// Transform Dataset to vector per channel
	for (size_t i = 0; i < NB_CHAN; ++i) { dataset[i].reserve(NB_SAMPLE * matrices.size()); }
	for (const auto& m : matrices) { for (size_t i = 0; i < NB_CHAN; ++i) { for (size_t j = 0; j < NB_SAMPLE; ++j) { dataset[i].push_back(m(i, j)); } } }

	// Sort and remove first (to begin by 0)
	for (auto& d : dataset)
	{
		std::sort(d.begin(), d.end());
		const auto first = d[0];
		for (auto& e : d) { e -= first; }
	}

	//========== Create Ref ==========
	const std::vector<std::vector<size_t>> ref =
	{
		{ 12, 10, 0, 15, 15, 0, 6, 12, 0, 11, 11, 0, 11, 14, 3 },
		{ 17, 0, 26, 0, 0, 18, 0, 28, 0, 0, 15, 0, 7, 0, 9 },
		{ 36, 0, 0, 34, 0, 0, 0, 0, 0, 15, 0, 0, 20, 0, 15 }
	};

	//========== Test ==========
	for (size_t i = 0; i < NB_CHAN; ++i)
	{
		const std::vector<size_t> hist = BinHist(dataset[i], 15);
		EXPECT_TRUE(isAlmostEqual(hist, ref[i])) << ErrorMsg("Bin Histogramm", hist, ref[i]);
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Misc, fitDistribution)
{
	const std::vector<Eigen::MatrixXd> matrices = Vector2DTo1D(InitDataset::Dataset());
	std::vector<std::vector<double>> dataset(NB_CHAN);
	// Transform Dataset to vector per channel
	for (size_t i = 0; i < NB_CHAN; ++i) { dataset[i].reserve(NB_SAMPLE * matrices.size()); }
	for (const auto& m : matrices) { for (size_t i = 0; i < NB_CHAN; ++i) { for (size_t j = 0; j < NB_SAMPLE; ++j) { dataset[i].push_back(m(i, j)); } } }

	// Begin Fit Distribution
	std::vector<double> mu(NB_CHAN), sigma(NB_CHAN);
	const std::vector<double> refMu    = { -0.840258269642149, - 2.10169835819046, 0.898301641809541 },
							  refSigma = { 2.76541902273525, 0.435493584265319, 0.435493584265319 };

	for (size_t i = 0; i < NB_CHAN; ++i)
	{
		FitDistribution(dataset[i], mu[i], sigma[i]);
		EXPECT_TRUE(isAlmostEqual(mu[i], refMu[i])) << ErrorMsg("Fit Distribution Mu", mu[i], refMu[i]);
		EXPECT_TRUE(isAlmostEqual(sigma[i], refSigma[i])) << ErrorMsg("Fit Distribution Sigma", sigma[i], refSigma[i]);
	}
}
//---------------------------------------------------------------------------------------------------
