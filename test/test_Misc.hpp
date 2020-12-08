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
#include "Init.hpp"
#include "misc.hpp"

#include <geometry/Misc.hpp>
#include <geometry/Basics.hpp>

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
TEST_F(Tests_Misc, Double_Range)
{
	const std::vector<double> calc1 = Geometry::doubleRange(0, 10, 2), calc2        = Geometry::doubleRange(0, 10, 2, false),
							  calc3 = Geometry::doubleRange(0.15, 3.05, 0.5), calc4 = Geometry::doubleRange(0.15, 3.05, 0.5, false),
							  ref1  = { 0, 2, 4, 6, 8, 10 }, ref2                   = { 0, 2, 4, 6, 8 },
							  ref3  = { 0.15, 0.65, 1.15, 1.65, 2.15, 2.65 }, ref4  = { 0.15, 0.65, 1.15, 1.65, 2.15, 2.65 };

	EXPECT_TRUE(isAlmostEqual(ref1, calc1)) << ErrorMsg("Double closed Range with integer value", ref1, calc1);
	EXPECT_TRUE(isAlmostEqual(ref2, calc2)) << ErrorMsg("Double opened Range with integer value", ref2, calc2);
	EXPECT_TRUE(isAlmostEqual(ref3, calc3)) << ErrorMsg("Double closed Range with double value", ref3, calc3);
	EXPECT_TRUE(isAlmostEqual(ref4, calc4)) << ErrorMsg("Double opened Range with double value", ref4, calc4);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Misc, Round_Index_Range)
{
	const std::vector<size_t> calc1 = Geometry::RoundIndexRange(0, 10, 2), calc2        = Geometry::RoundIndexRange(0, 10, 2, false),
							  calc3 = Geometry::RoundIndexRange(0.15, 3.15, 0.2), calc4 = Geometry::RoundIndexRange(0.15, 3.05, 0.2, false, false),
							  ref1  = { 0, 2, 4, 6, 8, 10 }, ref2                       = { 0, 2, 4, 6, 8 },
							  ref3  = { 0, 1, 2, 3 }, ref4                              = { 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3 };

	EXPECT_TRUE(isAlmostEqual(ref1, calc1)) << ErrorMsg("Round Index closed Range with integer value", ref1, calc1);
	EXPECT_TRUE(isAlmostEqual(ref2, calc2)) << ErrorMsg("Round Index opened Range with integer value", ref2, calc2);
	EXPECT_TRUE(isAlmostEqual(ref3, calc3)) << ErrorMsg("Round Index closed Range with double value", ref3, calc3);
	EXPECT_TRUE(isAlmostEqual(ref4, calc4)) << ErrorMsg("Round Index opened Range with double value", ref4, calc4);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Misc, Bin_Histogramm)
{
	//========== Create Dataset ==========
	const std::vector<Eigen::MatrixXd> matrices = Geometry::Vector2DTo1D(InitDataset::Dataset());
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
		const std::vector<size_t> hist = Geometry::BinHist(dataset[i], 15);
		EXPECT_TRUE(isAlmostEqual(hist, ref[i])) << ErrorMsg("Bin Histogramm", hist, ref[i]);
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Misc, Fit_Distribution)
{
	const std::vector<Eigen::MatrixXd> matrices = Geometry::Vector2DTo1D(InitDataset::Dataset());
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
		Geometry::FitDistribution(dataset[i], mu[i], sigma[i]);
		EXPECT_TRUE(isAlmostEqual(mu[i], refMu[i])) << ErrorMsg("Fit Distribution Mu", mu[i], refMu[i]);
		EXPECT_TRUE(isAlmostEqual(sigma[i], refSigma[i])) << ErrorMsg("Fit Distribution Sigma", sigma[i], refSigma[i]);
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Misc, Sorted_Eigen_Vector_Euclidian)
{
	std::vector<Eigen::MatrixXd> matrices   = Geometry::Vector2DTo1D(InitCovariance::LWF::Reference());
	const size_t n                          = matrices.size();
	std::vector<Eigen::MatrixXd> vectors    = InitEigenVector::Euclidian::Vectors();
	std::vector<std::vector<double>> values = InitEigenVector::Euclidian::Values();
	for (size_t i = 0; i < n; ++i)
	{
		Eigen::MatrixXd vec;
		std::vector<double> val;
		Geometry::sortedEigenVector(matrices[i], vec, val, Geometry::EMetric::Euclidian);
		EXPECT_TRUE(isAlmostEqual(vectors[i], vec)) << ErrorMsg("Eigen Vector sample " + std::to_string(i) + " : ", vectors[i], vec);
		EXPECT_TRUE(isAlmostEqual(values[i], val)) << ErrorMsg("Eigen Value sample " + std::to_string(i) + " : ", values[i], val);
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
TEST_F(Tests_Misc, Sorted_Eigen_Vector_Riemann)
{
	std::cout << "Not implemented" << std::endl;
	std::vector<Eigen::MatrixXd> matrices = Geometry::Vector2DTo1D(InitCovariance::LWF::Reference());
	const size_t n                        = matrices.size();
	//std::vector<Eigen::MatrixXd> vectors = InitEigenVector::Riemann::Vectors();
	//std::vector<std::vector<double>> values = InitEigenVector::Riemann::Values();
	for (size_t i = 0; i < n; ++i)
	{
		Eigen::MatrixXd vec;
		std::vector<double> val;
		Geometry::sortedEigenVector(matrices[i], vec, val, Geometry::EMetric::Riemann);
		//EXPECT_TRUE(isAlmostEqual(vectors[i], vec)) << ErrorMsg("Eigen Vector sample " + std::to_string(i) + " : ", vectors[i], vec);
		//EXPECT_TRUE(isAlmostEqual(values[i], val)) << ErrorMsg("Eigen Value sample " + std::to_string(i) + " : ", values[i], val);
	}
}
//---------------------------------------------------------------------------------------------------
