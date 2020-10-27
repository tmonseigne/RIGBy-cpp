#include "geometry/artifacts/CASR.hpp"

#include "geometry/Misc.hpp"
#include "geometry/Median.hpp"
#include "geometry/Covariance.hpp"
#include "geometry/Mean.hpp"

#include <boost/math/special_functions/detail/igamma_inverse.hpp>
#include <unsupported/Eigen/MatrixFunctions>

#include <cmath>
#include <numeric>

namespace Geometry {

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

///-------------------------------------------------------------------------------------------------
bool CASR::train(const std::vector<Eigen::MatrixXd>& dataset, const double rejectionLimit)
{
	if (dataset.empty() || dataset[0].size() == 0) { return false; }
	const size_t n = dataset.size();	// Number of samples
	m_nChannel     = dataset[0].rows();	// Number of channels

	//========== Compute the covariance matrix ==========
	std::vector<Eigen::MatrixXd> covs(n);
	//for (size_t i = 0; i < n; ++i) { if (!CovarianceMatrixLWF(dataset[i], covs[i])) { return false; } }	// We assume data is centered
	for (size_t i = 0; i < n; ++i) { if (!CovarianceMatrix(dataset[i], covs[i], EEstimator::LWF, EStandardization::Center)) { return false; } }

	//========== Compute Square Root of Median ==========
	if (!Median(covs, m_median)) { return false; }
	m_median = m_median.sqrt();

	//========== Compute Eigen vectors ==========
	Eigen::MatrixXd eigVector;
	std::vector<double> eigValues;
	sortedEigenVector(m_median, eigVector, eigValues, m_metric);

	//========== Compute the ponderate dataset ==========
	std::vector<Eigen::MatrixXd> newDataset;
	newDataset.reserve(n);
	for (const auto& m : dataset) { newDataset.push_back((m.transpose() * eigVector)); }	// Multiply by eigen vector (we transpose to have channels in column
	for (auto& m : newDataset) { m = m.cwiseProduct(m); }									// Square new signal

	//========== Compute the "fit" distribution ==========
	// Compute the RMS of each channel for each sample
	std::vector<std::vector<double>> rms(m_nChannel, std::vector<double>(n));
	for (size_t i = 0; i < n; ++i) { for (size_t j = 0; j < m_nChannel; ++j) { rms[j][i] = sqrt(newDataset[i].col(j).mean()); } }

	// Compute the "fit" distribution
	std::vector<double> mu(m_nChannel, 0.0), sigma(m_nChannel, 0.0);
	for (size_t i = 0; i < m_nChannel; ++i) { FitDistribution(rms[i], mu[i], sigma[i]); }

	// Compute the Transform Matrix
	m_treshold = Eigen::MatrixXd::Zero(m_nChannel, m_nChannel);
	for (size_t i = 0; i < m_nChannel; ++i) { m_treshold(i, i) = mu[i] + rejectionLimit * sigma[i]; }
	m_treshold *= eigVector.transpose();

	// Initialize Reconstruction matrix and trivial
	m_r       = Eigen::MatrixXd::Identity(m_nChannel, m_nChannel);
	m_trivial = true;
	return true;
}

bool CASR::process(const Eigen::MatrixXd& in, Eigen::MatrixXd& out)
{
	return false;
	// Actually eigen dependency version is too lower
	/*
	// Check if input data is compatible with train data and if we don't limit so mutch the reconstruction
	out = in;
	if (out.rows() != m_nChannel) { return false; }
	const size_t begin = size_t((1.0 - m_maxChannel) * m_nChannel);	// We define the number of channels to non reconstruct
	if (begin == m_nChannel) { return true; }
	if (m_r.size() == 0) { m_r = Eigen::MatrixXd::Identity(m_nChannel, m_nChannel); }

	// Compute Covariance matrix
	Eigen::MatrixXd cov;
	if (!CovarianceMatrix(in, cov, EEstimator::LWF, EStandardization::Center)) { return false; }
	if (m_cov.size() == 0) { m_cov = cov; }									// if first time
	else { if (!Mean({ m_cov, cov }, m_cov, m_metric)) { return false; } }	// else mean of the both

	// Compute Eigen vector & values
	Eigen::MatrixXd eigVector;
	std::vector<double> eigValues;
	sortedEigenVector(m_cov, eigVector, eigValues, m_metric);

	// Check if eigen values is over threshold computes during train (ponderate by eigen vector)
	Eigen::MatrixXd threshold = (m_treshold * eigVector).cwiseAbs2();
	bool trivial              = true;
	std::vector<bool> keep(m_nChannel, true);
	for (size_t i = begin; i < m_nChannel; ++i)
	{
		if (eigValues[i] >= threshold.col(i).sum())
		{
			keep[i] = false;
			trivial = false;
		}
	}

	// Check if All channels are clean
	if (trivial) { m_r = Eigen::MatrixXd::Identity(m_nChannel, m_nChannel); }
	else	// if not...
	{
		// Compute the reconstruction matrix with bad channels 
		Eigen::MatrixXd tmp = eigVector.transpose() * m_median;
		for (size_t i = begin; i < m_nChannel; ++i) { if (!keep[i]) { tmp.row(i).setZero(); } }
		const Eigen::MatrixXd newR = m_median * tmp.completeOrthogonalDecomposition().pseudoInverse() * eigVector.transpose();

		if (!m_trivial)
		{
			// Compute blend values for the samples
			const size_t nSample = in.cols();
			std::vector<double> blend(nSample);
			std::iota(blend.begin(), blend.end(), 1);	// Range 1 to nSample (inclued)
			for (auto& b : blend) { b = (1 - cos(M_PI * (b / double(nSample)))) / 2.0; }

			// Apply reconstruction ponderate by the blend (we considere the old reconstruction matrix for the second part)
			Eigen::MatrixXd t1 = newR * in;
			Eigen::MatrixXd t2 = m_r * in;
			for (size_t i = 0; i < nSample; ++i) { out.col(i) = (blend[i] * t1.col(i)) + ((1 - blend[i]) * t2.col(i)); }
		}
		m_r = newR;	// Update the reconstruction matrix
	}
	m_trivial = trivial;
	return true;
	*/
}
///-------------------------------------------------------------------------------------------------

//*****************************
//***** Override Operator *****
//*****************************

///-------------------------------------------------------------------------------------------------
bool CASR::isEqual(const CASR& obj, const double precision) const
{
	return m_metric == obj.m_metric && m_median == obj.m_median && m_treshold == obj.m_treshold;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CASR::copy(const CASR& obj)
{
	m_metric   = obj.m_metric;
	m_median   = obj.m_median;
	m_treshold = obj.m_treshold;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
std::string CASR::toString() const
{
	std::stringstream ss;
	ss << "Metric : " << (m_metric == EMetric::Riemann ? "Riemann" : "Euclidian") << std::endl;	// tostring(EMetrics) doesn't work
	if (m_nChannel == 0) { ss << "Train not done" << std::endl; }
	else
	{
		ss << "Train done." << std::endl;
		ss << size_t(m_maxChannel * double(m_nChannel)) << "/" << m_nChannel << " channels can be reconstruted." << std::endl;
		ss << "Median matrix is : " << std::endl << m_median << std::endl;
		ss << "Treshold matrix is : " << std::endl << m_treshold << std::endl;
		if (m_cov.size() == 0) { ss << "No process launched yet." << std::endl; }
		else
		{
			ss << "Last sample " << (m_trivial ? "was" : "wasn't") << " trivial." << std::endl;
			ss << "Last Reconstruction Matrix : " << std::endl << m_r << std::endl;
			ss << "Last Covariance Matrix : " << std::endl << m_cov << std::endl;
		}
	}
	return ss.str();
}
///-------------------------------------------------------------------------------------------------

}  // namespace Geometry
