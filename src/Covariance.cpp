# include "geometry/Covariance.hpp"
#include <algorithm>		// std::min/max

namespace Geometry {

//***********************************************************
//******************** COVARIANCES BASES ********************
//***********************************************************
//---------------------------------------------------------------------------------------------------
double Variance(const Eigen::RowVectorXd& x)
{
	const size_t S = x.cols();								// Number of Samples				=> S
	if (S == 0) { return 0; }								// If false input

	const double mu = x.mean();
	return x.cwiseProduct(x).sum() / S - mu * mu;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
double Covariance(const Eigen::RowVectorXd& x, const Eigen::RowVectorXd& y)
{
	const size_t xS = x.cols(), yS = y.cols();				// Number of Samples				=> S
	if (xS == 0 || xS != yS) { return 0; }					// If false input
	return (x.cwiseProduct(y).sum() - x.sum() * y.sum() / xS) / xS;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool ShrunkCovariance(Eigen::MatrixXd& cov, const double shrinkage)
{
	if (!InRange(shrinkage, 0, 1)) { return false; }		// Verification
	const size_t n = cov.rows();							// Number of Features				=> N

	const double coef = shrinkage * cov.trace() / n;		// Diagonal Coefficient
	cov               = (1 - shrinkage) * cov;				// Shrinkage
	for (size_t i = 0; i < n; ++i) { cov(i, i) += coef; }	// Add Diagonal Coefficient

	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool ShrunkCovariance(const Eigen::MatrixXd& in, Eigen::MatrixXd& out, const double shrinkage)
{
	out = in;
	return ShrunkCovariance(out, shrinkage);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool CovarianceMatrix(const Eigen::MatrixXd& in, Eigen::MatrixXd& out, const EEstimator estimator, const EStandardization standard)
{
	if (!IsNotEmpty(in)) { return false; }					// Verification
	Eigen::MatrixXd sample;
	MatrixStandardization(in, sample, standard);			// Standardization
	switch (estimator)										// Switch Method
	{
		case EEstimator::COV: return CovarianceMatrixCOV(sample, out);
		case EEstimator::SCM: return CovarianceMatrixSCM(sample, out);
		case EEstimator::LWF: return CovarianceMatrixLWF(sample, out);
		case EEstimator::OAS: return CovarianceMatrixOAS(sample, out);
		case EEstimator::MCD: return CovarianceMatrixMCD(sample, out);
		case EEstimator::COR: return CovarianceMatrixCOR(sample, out);
		default: return CovarianceMatrixIDE(sample, out);
	}
}
//---------------------------------------------------------------------------------------------------

//***********************************************************
//***********************************************************
//***********************************************************

//***********************************************************
//******************** COVARIANCES TYPES ********************
//***********************************************************
//---------------------------------------------------------------------------------------------------
bool CovarianceMatrixCOV(const Eigen::MatrixXd& samples, Eigen::MatrixXd& cov)
{
	const size_t n = samples.rows();						// Number of Features				=> N

	cov.resize(n, n);										// Init size of matrix
	for (size_t i = 0; i < n; ++i)
	{
		const Eigen::RowVectorXd ri = samples.row(i);
		cov(i, i)                   = Variance(ri);			// Diagonal Value

		for (size_t j = i + 1; j < n; ++j)
		{
			const Eigen::RowVectorXd rj = samples.row(j);
			cov(i, j)                   = cov(j, i) = Covariance(ri, rj);	// Symetric covariance
		}
	}
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool CovarianceMatrixSCM(const Eigen::MatrixXd& samples, Eigen::MatrixXd& cov)
{
	cov = samples * samples.transpose();					// X*X^T
	cov /= cov.trace();										// X*X^T / trace(X*X^T)
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool CovarianceMatrixLWF(const Eigen::MatrixXd& samples, Eigen::MatrixXd& cov)
{
	const size_t n = samples.rows(), S = samples.cols();	// Number of Features & Samples		=> N & S

	CovarianceMatrixCOV(samples, cov);						// Initial Covariance Matrix		=> Cov
	const double mu        = cov.trace() / n;
	Eigen::MatrixXd mDelta = cov;							// mDelta = cov - mu * I_n
	for (size_t i = 0; i < n; ++i) { mDelta(i, i) -= mu; }
	const Eigen::MatrixXd x2   = samples.cwiseProduct(samples),	// Squared each sample				=> X^2
						  cov2 = cov.cwiseProduct(cov);			// Squared each element of Cov		=> Cov^2

	const double delta     = mDelta.cwiseProduct(mDelta).sum() / n,
				 beta      = 1. / double(n * S) * (x2 * x2.transpose() / double(S) - cov2).sum(),
				 shrinkage = std::min(beta, delta) / delta;	// Assure shrinkage <= 1

	return ShrunkCovariance(cov, shrinkage);				// Shrinkage of the matrix
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool CovarianceMatrixOAS(const Eigen::MatrixXd& samples, Eigen::MatrixXd& cov)
{
	const size_t n = samples.rows(), S = samples.cols();	// Number of Features & Samples		=> N & S
	CovarianceMatrixCOV(samples, cov);						// Initial Covariance Matrix		=> Cov

	// Compute Shrinkage : Formula from Chen et al.'s
	const double mu        = cov.trace() / n,
				 mu2       = mu * mu,
				 alpha     = cov.cwiseProduct(cov).mean(),
				 num       = alpha + mu2,
				 den       = (S + 1) * (alpha - mu2 / n),
				 shrinkage = (den == 0) ? 1.0 : std::min(num / den, 1.0);

	return ShrunkCovariance(cov, shrinkage);				// Shrinkage of the matrix
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool CovarianceMatrixMCD(const Eigen::MatrixXd& samples, Eigen::MatrixXd& cov) { return CovarianceMatrixIDE(samples, cov); }
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool CovarianceMatrixCOR(const Eigen::MatrixXd& samples, Eigen::MatrixXd& cov)
{
	const size_t n = samples.rows();						// Number of Features				=> N
	CovarianceMatrixCOV(samples, cov);						// Initial Covariance Matrix		=> Cov
	const Eigen::MatrixXd d = cov.diagonal().cwiseSqrt();	// Squared root of diagonal

	for (size_t i = 0; i < n; ++i) { for (size_t j = 0; j < n; ++j) { cov(i, j) /= d(i) * d(j); } }
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool CovarianceMatrixIDE(const Eigen::MatrixXd& samples, Eigen::MatrixXd& cov)
{
	cov = Eigen::MatrixXd::Identity(samples.rows(), samples.rows());
	return true;
}
//---------------------------------------------------------------------------------------------------
//***********************************************************
//***********************************************************
//***********************************************************

}  // namespace Geometry
