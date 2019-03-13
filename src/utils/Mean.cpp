#include "Metrics.hpp"
#include "Basics.hpp"
#include "Geodesic.hpp"
#include "Distance.hpp"
#include "Mean.hpp"
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>

using namespace Eigen;
using namespace std;

//const double EPSILON = 0.000000001;	// 10^{-9}
const double EPSILON = 0.0001;			// 10^{-4}
const size_t ITER_MAX = 50;

//---------------------------------------------------------------------------------------------------
bool Mean(const std::vector<MatrixXd>& covs, MatrixXd& mean, const EMetrics metric)
{
	if (covs.empty()) { return false; }							// If no matrix in vector
	// Force Square Matrix for non Euclidian and non Identity metric
	if (!areSquare(covs) && (metric != Metric_Euclidian && metric != Metric_Identity))
	{
		cerr << "Non Square Matrix is invalid with " << MetricToString(metric) << " metric." << endl;
		return false;
	}
	if (covs.size() == 1) { mean = covs[0];		return true; }	// If just one matrix in vector

	switch (metric)												// Switch method
	{
		case Metric_Riemann: return MeanRiemann(covs, mean);
		case Metric_Euclidian: return MeanEuclidian(covs, mean);
		case Metric_LogEuclidian: return MeanLogEuclidian(covs, mean);
		case Metric_LogDet: return MeanLogDet(covs, mean);
		case Metric_Kullback: return MeanKullback(covs, mean);
		case Metric_ALE: return MeanALE(covs, mean);
		case Metric_Harmonic: return MeanHarmonic(covs, mean);
		case Metric_Wasserstein: return MeanWasserstein(covs, mean);
		case Metric_Identity:
		default: return MeanIdentity(covs, mean);
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool AJDPham(const std::vector<MatrixXd>& covs, MatrixXd& ajd, double /*epsilon*/, const int /*maxIter*/)
{
	MeanIdentity(covs, ajd);
	return false;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MeanRiemann(const vector<MatrixXd>& covs, MatrixXd& mean)
{
	const size_t k = covs.size(), n = covs[0].rows();			// Number of Matrix & Features		=> K & N
	size_t i = 0;												// Index of Covariance Matrix		=> i
	double nu = 1.0,											// Coefficient change				=> nu
		   tau = numeric_limits<double>::max(),					// Coefficient change criterion		=> tau
		   crit = numeric_limits<double>::max();				// Current change					=> crit
	if (!MeanEuclidian(covs, mean)) { return false; }			// Initial Mean

	while (i < ITER_MAX && EPSILON < crit && EPSILON < nu)		// Stopping criterion
	{
		i++;													// Iteration Criterion
		const MatrixXd sC = mean.sqrt(), isC = sC.inverse();	// Square root & Inverse Square root of Mean	=> sC & isC
		MatrixXd mJ = MatrixXd::Zero(n, n);						// Change							=> J
		for (const MatrixXd& cov : covs) { mJ += (isC * cov * isC).log(); }	// Sum of log(isC*Ci*isC)
		mJ /= double(k);										// Normalization
		crit = mJ.norm();										// Current change criterion
		mean = sC * (nu * mJ).exp() * sC;						// Update Mean		=> M = sC * exp(nu*J) * sC

		const double h = nu * crit;								// Update Coefficient change
		if (h < tau)
		{
			nu *= 0.95;
			tau = h;
		}
		else { nu *= 0.5; }
	}
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MeanEuclidian(const vector<MatrixXd>& covs, MatrixXd& mean)
{
	const size_t k = covs.size(), n = covs[0].rows();			// Number of Matrix & Features		=> K & N
	mean = MatrixXd::Zero(n, n);								// Initial Mean
	for (const MatrixXd& cov : covs) { mean += cov; }			// Sum of Ci
	mean /= double(k);											// Normalization
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MeanLogEuclidian(const vector<MatrixXd>& covs, MatrixXd& mean)
{
	const size_t k = covs.size(), n = covs[0].rows();			// Number of Matrix & Features		=> K & N
	mean = MatrixXd::Zero(n, n);								// Initial Mean
	for (const MatrixXd& cov : covs) { mean += cov.log(); }		// Sum of log(Ci)
	mean = (mean / double(k)).exp();							// Normalization
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MeanLogDet(const vector<MatrixXd>& covs, MatrixXd& mean)
{
	const size_t k = covs.size(), n = covs[0].rows();			// Number of Matrix & Features		=> K & N
	size_t i = 0;												// Index of Covariance Matrix		=> i
	double crit = std::numeric_limits<double>::max();			// Current change					=> crit
	if (!MeanEuclidian(covs, mean)) { return false; }			// Initial Mean

	while (i < ITER_MAX && EPSILON < crit)						// Stopping criterion
	{
		i++;													// Iteration Criterion
		MatrixXd mJ = MatrixXd::Zero(n, n);						// Change							=> J

		for (const MatrixXd& cov : covs) { mJ += (0.5 * (cov + mean)).inverse(); }	// Sum of ((Ci+M)/2)^{-1}
		mJ = (mJ / double(k)).inverse();						// Normalization
		crit = (mJ - mean).norm();								// Current change criterion
		mean = mJ;												// Update mean
	}
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MeanKullback(const vector<MatrixXd>& covs, MatrixXd& mean)
{
	MatrixXd m1, m2;
	if (!MeanEuclidian(covs, m1)) { return false; }
	if (!MeanHarmonic(covs, m2)) { return false; }
	if (!GeodesicRiemann(m1, m2, mean, 0.5)) { return false; }
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MeanWasserstein(const vector<MatrixXd>& covs, MatrixXd& mean)
{
	const size_t k = covs.size(), n = covs[0].rows();			// Number of Matrix & Features		=> K & N
	size_t i = 0;												// Index of Covariance Matrix		=> i
	double crit = std::numeric_limits<double>::max();			// Current change					=> crit

	if (!MeanEuclidian(covs, mean)) { return false; }			// Initial Mean
	MatrixXd sC = mean.sqrt();									// Square root of Mean				=> sC

	while (i < ITER_MAX && EPSILON < crit)						// Stopping criterion
	{
		i++;													// Iteration Criterion
		MatrixXd mJ = MatrixXd::Zero(n, n);						// Change							=> J

		for (const MatrixXd& cov : covs) { mJ += (sC * cov * sC).sqrt(); }	// Sum of sqrt(sC*Ci*sC)
		mJ /= double(k);										// Normalization

		const MatrixXd sJ = mJ.sqrt();							// Square root of change			=> sJ
		crit = (sJ - sC).norm();								// Current change criterion
		sC = sJ;												// Update sC
	}
	mean = sC * sC;												// Un-square root 
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MeanALE(const vector<MatrixXd>& covs, MatrixXd& mean)
{
	const size_t k = covs.size(), n = covs[0].rows();			// Number of Matrix & Features		=> K & N
	size_t i = 0;												// Index of Covariance Matrix		=> i
	double crit = std::numeric_limits<double>::max();			// Change criterion					=> crit
	if (!AJDPham(covs, mean)) { return false; }					// Initial Mean
	MatrixXd mJ;												// Change

	while (i < ITER_MAX && EPSILON < crit)						// Stopping criterion
	{
		i++;													// Iteration Criterion
		mJ = MatrixXd::Zero(n, n);								// Change							=> J

		for (const MatrixXd& cov : covs) { mJ += (mean.transpose() * cov * mean).log(); }	// Sum of log(C^T*Ci*C)
		mJ /= double(k);										// Normalization

		MatrixXd update = mJ.exp().diagonal().asDiagonal();		// Update Form						=> U
		mean = mean * update.sqrt().inverse();					// Update Mean M = M * U^{-1/2}

		crit = DistanceRiemann(MatrixXd::Identity(n, n), update);
	}

	mJ = MatrixXd::Zero(n, n);									// Last Change						=> J
	for (const MatrixXd& cov : covs) { mJ += (mean.transpose() * cov * mean).log(); }	// Sum of log(C^T*Ci*C)
	mJ /= double(k);											// Normalization

	MatrixXd mA = mean.inverse();
	mean = mA.transpose() * mJ.exp() * mA;
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MeanHarmonic(const vector<MatrixXd>& covs, MatrixXd& mean)
{
	const size_t k = covs.size(), n = covs[0].rows();			// Number of Matrix & Features		=> K & N
	mean = MatrixXd::Zero(n, n);								// Initial Mean
	for (const MatrixXd& cov : covs) { mean += cov.inverse(); }	// Sum of Inverse
	mean = (mean / double(k)).inverse();						// Normalization and inverse
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MeanIdentity(const vector<MatrixXd>& covs, MatrixXd& mean)
{
	mean = MatrixXd::Identity(covs[0].rows(), covs[0].cols());
	return true;
}
//---------------------------------------------------------------------------------------------------
