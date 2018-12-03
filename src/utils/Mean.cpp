#include "Metrics.hpp"
#include "Basics.hpp"
#include "Geodesic.hpp"
#include "Distance.hpp"
#include "Mean.hpp"
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;
using namespace std;

//const double EPSILON = 0.000000001;	// 10^{-9}
const double EPSILON = 0.0001;			// 10^{-4}
const size_t ITER_MAX = 50;


bool Mean(const std::vector<MatrixXd>& covs, MatrixXd& mean, const EMetrics metric)
{
	if (!areSquare(covs)) { return false; }
	if (covs.size() == 1)
	{
		mean = covs[0];
		return true;
	}

	switch (metric)
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

bool AJDPham(const std::vector<MatrixXd>& covs, MatrixXd& ajd, double epsilon, const int maxIter)
{
	(void)epsilon;
	(void)maxIter;
	MeanIdentity(covs, ajd);
	return false;
}
//---------------------------------------------------------------------------------------------------

bool MeanRiemann(const vector<MatrixXd>& covs, MatrixXd& mean)
{
	const size_t k = covs.size(),							// Number of Covariance Matrix		=> K
				 n = covs[0].rows();						// Number of Features				=> N
	size_t i = 0;											// Index of Covariance Matrix		=> i
	double nu = 1.0,										// Coefficient change				=> nu
		   tau = numeric_limits<double>::max(),				// Coefficient change criterion		=> tau
		   crit = numeric_limits<double>::max();			// Current change					=> crit
	if (!MeanEuclidian(covs, mean)) { return false; }		// Initial Mean

	while (i < ITER_MAX && EPSILON < crit && EPSILON < nu)
	{
		i++;
		const MatrixXd sC = mean.sqrt(),					// Square root of Mean				=> sC
					   isC = sC.inverse();					// Inverse Square root of Mean		=> isC
		MatrixXd mJ = MatrixXd::Zero(n, n);					// Change							=> J
		for (const MatrixXd& cov : covs)					// Sum of log(isC*Ci*isC)
		{
			mJ += (isC * cov * isC).log();
		}
		mJ /= double(k);									// Normalization
		crit = mJ.norm();									// Current change criterion
		mean = sC * (nu * mJ).exp() * sC;					// Update Mean M = sC * exp(nu*J) * sC

		const double h = nu * crit;							// Update Coefficient change
		if (h < tau)
		{
			nu *= 0.95;
			tau = h;
		}
		else
		{
			nu *= 0.5;
		}
	}
	return true;
}
//---------------------------------------------------------------------------------------------------

bool MeanEuclidian(const vector<MatrixXd>& covs, MatrixXd& mean)
{
	const size_t k = covs.size(),							// Number of Covariance Matrix		=> K
				 n = covs[0].rows();						// Number of Features				=> N
	mean = MatrixXd::Zero(n, n);							// Initial Mean
	for (const MatrixXd& cov : covs) { mean += cov; }		// Sum of Ci
	mean /= double(k);										// Normalization
	return true;
}
//---------------------------------------------------------------------------------------------------
bool MeanLogEuclidian(const vector<MatrixXd>& covs, MatrixXd& mean)
{
	const size_t k = covs.size(),							// Number of Covariance Matrix		=> K
				 n = covs[0].rows();						// Number of Features				=> N
	mean = MatrixXd::Zero(n, n);							// Initial Mean
	for (const MatrixXd& cov : covs) { mean += cov.log(); }	// Sum of log(Ci)
	mean = (mean / double(k)).exp();						// Normalization
	return true;
}
//---------------------------------------------------------------------------------------------------
bool MeanLogDet(const vector<MatrixXd>& covs, MatrixXd& mean)
{
	const size_t k = covs.size(),							// Number of Covariance Matrix		=> K
				 n = covs[0].rows();						// Number of Features				=> N
	size_t i = 0;											// Index of Covariance Matrix		=> i
	double crit = std::numeric_limits<double>::max();		// Current change					=> crit
	if (!MeanEuclidian(covs, mean)) { return false; }		// Initial Mean

	while (i < ITER_MAX && EPSILON < crit)
	{
		i++;
		MatrixXd mJ = MatrixXd::Zero(n, n);					// Change							=> J

		for (const MatrixXd& cov : covs)					// Sum of ((Ci+M)/2)^{-1}
		{
			mJ += (0.5 * (cov + mean)).inverse();
		}
		mJ = (mJ / double(k)).inverse();					// Normalization
		crit = (mJ - mean).norm();							// Current change criterion
		mean = mJ;											// Update mean
	}
	return true;
}
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

bool MeanWasserstein(const vector<MatrixXd>& covs, MatrixXd& mean)
{
	const size_t k = covs.size(),							// Number of Covariance Matrix		=> K
				 n = covs[0].rows();						// Number of Features				=> N
	size_t i = 0;											// Index of Covariance Matrix		=> i
	double crit = std::numeric_limits<double>::max();		// Current change					=> crit

	if (!MeanEuclidian(covs, mean)) { return false; }		// Initial Mean
	MatrixXd sC = mean.sqrt();								// Square root of Mean				=> sC

	while (i < ITER_MAX && EPSILON < crit)
	{
		i++;
		MatrixXd mJ = MatrixXd::Zero(n, n);					// Change							=> J

		for (const MatrixXd& cov : covs)					// Sum of sqrt(sC*Ci*sC)
		{
			mJ += (sC * cov * sC).sqrt();
		}
		mJ /= double(k);									// Normalization

		const MatrixXd sJ = mJ.sqrt();						// Square root of change			=> sJ
		crit = (sJ - sC).norm();							// Current change criterion
		sC = sJ;											// Update sC
	}
	mean = sC * sC;											// Un-square root 
	return true;
}
//---------------------------------------------------------------------------------------------------

bool MeanALE(const vector<MatrixXd>& covs, MatrixXd& mean)
{
	const size_t k = covs.size(),							// Number of Covariance Matrix		=> K
				 n = covs[0].rows();						// Number of Features				=> N
	size_t i = 0;											// Index of Covariance Matrix		=> i
	double crit = std::numeric_limits<double>::max();		// Change criterion					=> crit
	if (!AJDPham(covs, mean)) { return false; }				// Initial Mean
	MatrixXd mJ;											// Change

	while (i < ITER_MAX && EPSILON < crit)
	{
		i++;
		mJ = MatrixXd::Zero(n, n);							// Change							=> J

		for (const MatrixXd& cov : covs)					// Sum of log(C^T*Ci*C)
		{
			mJ += (mean.transpose() * cov * mean).log();
		}
		mJ /= double(k);									// Normalization

		MatrixXd update = mJ.exp().diagonal().asDiagonal();	// Update Form						=> U
		mean = mean * update.sqrt().inverse();				// Update Mean M = M * U^{-1/2}

		crit = DistanceRiemann(MatrixXd::Identity(n, n), update);
	}

	mJ = MatrixXd::Zero(n, n);
	for (const MatrixXd& cov : covs)
	{
		mJ += (mean.transpose() * cov * mean).log();
	}
	mJ /= double(k);

	MatrixXd mA = mean.inverse();
	mean = mA.transpose() * mJ.exp() * mA;
	return true;
}
//---------------------------------------------------------------------------------------------------

bool MeanHarmonic(const vector<MatrixXd>& covs, MatrixXd& mean)
{
	const size_t k = covs.size(),							// Number of Covariance Matrix		=> K
				 n = covs[0].rows();						// Number of Features				=> N
	mean = MatrixXd::Zero(n, n);							// Initial Mean
	for (const MatrixXd& cov : covs)						// Sum of Inverse
	{
		mean += cov.inverse();
	}
	mean = (mean / double(k)).inverse();					// Normalization and inverse
	return true;
}
//---------------------------------------------------------------------------------------------------

bool MeanIdentity(const vector<MatrixXd>& covs, MatrixXd& mean)
{
	mean = MatrixXd::Identity(covs[0].rows(), covs[0].rows());
	return true;
}
//---------------------------------------------------------------------------------------------------
