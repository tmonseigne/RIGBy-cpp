#include "geometry/Median.hpp"

#include <iostream>

#include "geometry/Basics.hpp"
#include "geometry/Featurization.hpp"
#include "geometry/Mean.hpp"

namespace Geometry {

//---------------------------------------------------------------------------------------------------
double Median(const Eigen::MatrixXd& m)
{
	const std::vector<double> v(m.data(), m.data() + m.rows() * m.cols());
	return Median(v);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool Median(const std::vector<Eigen::MatrixXd>& matrices, Eigen::MatrixXd& median, const double epsilon, const size_t maxIter, const EMetric& metric)
{
	if (matrices.empty()) { return false; }						// If no matrix in vector
	if (matrices.size() == 1)									// If just one matrix in vector
	{
		median = matrices[0];
		return true;
	}
	if (!HaveSameSize(matrices))								// If different sizes
	{
		std::cout << "Matrices haven't same size." << std::endl;
		return false;
	}
	if (!IsSquare(matrices[0]) && metric == EMetric::Riemann)	// If non square for Riemann metric
	{
		std::cout << "Non Square Matrix is invalid with " << toString(metric) << " metric." << std::endl;
		return false;
	}

	switch (metric)
	{
		case EMetric::Riemann: return MedianRiemann(matrices, median, epsilon, maxIter);
		case EMetric::Euclidian: return MedianEuclidian(matrices, median, epsilon, maxIter);
		case EMetric::Identity: return MedianIdentity(matrices, median);
		case EMetric::LogEuclidian:
		case EMetric::LogDet:
		case EMetric::Kullback:
		case EMetric::ALE:
		case EMetric::Harmonic:
		case EMetric::Wasserstein:
			std::cout << toString(metric) << " metric not implemented." << std::endl;
			return false;
	}
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MedianEuclidian(const std::vector<Eigen::MatrixXd>& matrices, Eigen::MatrixXd& median, const double epsilon, const size_t maxIter)
{
	if (matrices.empty() || matrices[0].size() == 0) { return false; }
	const size_t n = matrices.size();					// Number of sample

	// Initial Median is the median of each channel in all matrix of dataset
	median = matrices[0];								// to copy size
	for (size_t i = 0; i < size_t(median.size()); ++i)
	{
		std::vector<double> tmp;
		tmp.reserve(n);									// Reserve to optimize (a little) the pushback memory access.
		for (const auto& cov : matrices) { tmp.push_back(cov.data()[i]); }	// Stack value number i of all matrix
		median.data()[i] = Median(tmp);
	}

	size_t iter = 0;									// number of iteration
	double gain = epsilon;								// Gain since last compute
	while (iter < maxIter && gain >= epsilon)
	{
		Eigen::MatrixXd prev = median;					// Keep old median
		median.setZero();								// Reset median
		double sumCoefs = 0;							// Sum of Coefficient
		for (const auto& cov : matrices)
		{
			//Eigen::MatrixXd difference = cov - prev;
			//double coef = sqrt(difference.cwiseProduct(difference).sum());
			if (cov.isApprox(prev)) { continue; }		// In this case, Median is exactly this current matrix so we don't consider this matrix
			double coef = (cov - prev).norm();
			// Personnal hack and security
			coef = 1.0 / coef;
			sumCoefs += coef;							// Sum for normalization
			median += coef * cov;						// Add to the new median
		}
		if (sumCoefs > 0.0) { median /= sumCoefs; }		// Normalize

		gain = (median - prev).norm() / median.norm();	// It's the Frobenius norm
		iter++;
	}
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MedianRiemann(const std::vector<Eigen::MatrixXd>& matrices, Eigen::MatrixXd& median, const double epsilon, const size_t maxIter)
{
	if (matrices.empty() || !IsSquare(matrices[0])) { return false; }
	const size_t n  = matrices.size();						// Number of sample
	const size_t nf = matrices[0].rows() * (matrices[0].rows() + 1) / 2;	// Number of Features in tangent space
	size_t iter     = 0;									// number of iteration
	if (!MeanEuclidian(matrices, median)) { return false; }	// Initialize Median

	double gain = epsilon;									// Gain since last compute
	std::vector<Eigen::MatrixXd> mats;
	mats.reserve(n);
	for (const auto& m : matrices) { mats.push_back(m); }
	while (iter < maxIter)
	{
		// Compute Tangent space of all matrices & sum of euclidian distance of each transposed matrix
		std::vector<Eigen::RowVectorXd> ts(n);
		double sum = 0.0;
		for (size_t i = 0; i < n; ++i)
		{
			if (!TangentSpace(mats[i], ts[i], median)) { return false; }
			sum += sqrt(ts[i].cwiseAbs2().sum());
		}
		if (std::abs((sum - gain) / gain) < epsilon) { break; }	// std::abs call fabs to keep type

		// Arithmetic median in tangent space
		std::vector<std::vector<double>> transposeTs(nf, std::vector<double>(n));
		Eigen::RowVectorXd featureMedian(nf);
		for (size_t i = 0; i < n; ++i) { for (size_t j = 0; j < nf; ++j) { transposeTs[j][i] = ts[i][j]; } }
		for (size_t j = 0; j < nf; ++j) { featureMedian[j] = Median(transposeTs[j]); }

		// back to the manifold
		Eigen::MatrixXd tmp;
		if (!UnTangentSpace(featureMedian, tmp, median)) { return false; }
		gain   = sum;										// Update gain
		median = tmp;										// Update Median
		iter++;
	}
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MedianIdentity(const std::vector<Eigen::MatrixXd>& matrices, Eigen::MatrixXd& median)
{
	median = Eigen::MatrixXd::Identity(matrices[0].rows(), matrices[0].cols());
	return true;
}
//---------------------------------------------------------------------------------------------------


}  // namespace Geometry
