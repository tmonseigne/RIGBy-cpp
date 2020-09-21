#include "CASR.hpp"

#include "utils/Misc.hpp"
#include "utils/Covariance.hpp"

#include <boost/math/special_functions/gamma.hpp>
#include <unsupported/Eigen/MatrixFunctions>

#include <cmath>
#include <numeric>

///-------------------------------------------------------------------------------------------------
bool CASR::train(const std::vector<Eigen::MatrixXd>& dataset, const double rejectionLimit)
{
	if (dataset.empty() || dataset[0].size() == 0) { return false; }
	const size_t n = dataset.size();	// Number of samples
	const size_t c = dataset[0].rows();	// Number of channels

	//========== Compute the covariance matrix ==========
	std::vector<Eigen::MatrixXd> covs(n);
	//for (size_t i = 0; i < n; ++i) { if (!CovarianceMatrixLWF(dataset[i], covs[i])) { return false; } }	// We assume data is centered
	for (size_t i = 0; i < n; ++i)
	{
		if (!CovarianceMatrix(dataset[i], covs[i], EEstimator::LWF, EStandardization::Center)) { return false; }
	}

	//========== Compute Square Root of Median ==========
	if (!Median(covs, m_median)) { return false; }
	m_median = m_median.sqrt();

	//========== Compute Eigen vectors ==========
	//Eigen::MatrixXd eigVector;
	//std::vector<double> eigValues;
	//if (m_metric == EMetric::Riemann) { if (!RiemannianNonLinearEigenVector(median, eigVector)) { return false; } }
	//else
	//{
	//Eigen::EigenSolver<Eigen::MatrixXd> es(median);
	//eigVector                 = es.eigenvectors().real();	// It's complex by default but all imaginary part are 0
	//const Eigen::MatrixXd tmp = es.eigenvalues().real();	// It's complex by default but all imaginary part are 0
	//eigValues                 = std::vector<double>(tmp.data(), tmp.data() + tmp.size());
	//}

	const Eigen::EigenSolver<Eigen::MatrixXd> es(m_median);
	Eigen::MatrixXd eigVector     = es.eigenvectors().real();	// It's complex by default but all imaginary part are 0
	const Eigen::MatrixXd tmp     = es.eigenvalues().real();	// It's complex by default but all imaginary part are 0
	std::vector<double> eigValues = std::vector<double>(tmp.data(), tmp.data() + tmp.size());

	// Sort Eigen vector
	std::vector<size_t> idx(eigValues.size());
	std::iota(idx.begin(), idx.end(), 0);
	std::stable_sort(idx.begin(), idx.end(), [&eigValues](const size_t i1, const size_t i2) { return eigValues[i1] < eigValues[i2]; });

	Eigen::MatrixXd tmpEigVector = eigVector;
	for (size_t i = 0; i < eigVector.cols(); ++i) { eigVector.col(i) = tmpEigVector.col(idx[i]); }

	//========== Compute the ponderate dataset ==========
	std::vector<Eigen::MatrixXd> newDataset;
	newDataset.reserve(n);
	for (const auto& m : dataset) { newDataset.push_back((m.transpose() * eigVector)); }	// Multiply by eigen vector (we transpose to have channels in column
	for (auto& m : newDataset) { m = m.cwiseProduct(m); }									// Square new signal

	//========== Compute the "fit" distribution ==========
	// Compute the RMS of each channel for each sample
	std::vector<std::vector<double>> rms(c, std::vector<double>(n));
	for (size_t i = 0; i < n; ++i) { for (size_t j = 0; j < c; ++j) { rms[j][i] = sqrt(newDataset[i].col(j).mean()); } }

	// Compute the "fit" distribution
	std::vector<double> mu(c, 0.0), sigma(c, 0.0);
	for (size_t i = 0; i < c; ++i) { FitDistribution(rms[i], mu[i], sigma[i]); }

	// Compute the Transform Matrix
	m_transform = Eigen::MatrixXd::Zero(c, c);
	for (size_t i = 0; i < c; ++i) { m_transform(i, i) = mu[i] + rejectionLimit * sigma[i]; }
	m_transform *= eigVector.transpose();
	return true;
}
///-------------------------------------------------------------------------------------------------

//*****************************
//***** Override Operator *****
//*****************************

///-------------------------------------------------------------------------------------------------
bool CASR::isEqual(const CASR& obj, const double precision) const
{
	return m_metric == obj.m_metric && m_median == obj.m_median && m_transform == obj.m_transform;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CASR::copy(const CASR& obj)
{
	m_metric = obj.m_metric;
	m_median = obj.m_median;
	m_transform = obj.m_transform;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
std::string CASR::toString() const
{
	std::stringstream ss;
	ss << "Metric = " << (m_metric == EMetric::Riemann ? "Riemann" : "Euclidian") << std::endl;	// tostring(EMetrics) doesn't work
	ss << "Median = " << std::endl << m_median << std::endl;
	ss << "Transformation matrix = " << std::endl << m_transform << std::endl;
	return ss.str();
}
///-------------------------------------------------------------------------------------------------
