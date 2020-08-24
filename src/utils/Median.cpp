#include "Median.hpp"
#include "Featurization.hpp"

//---------------------------------------------------------------------------------------------------
double Median(const Eigen::MatrixXd& m)
{
	std::vector<double> v(m.data(), m.data() + m.rows() * m.cols());
	return Median(v);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool Median(const std::vector<Eigen::MatrixXd>& covs, Eigen::MatrixXd& median, const double epsilon, const int maxIter)
{
	if (covs.empty() || covs[0].size() == 0) { return false; }
	const size_t n = covs.size();		// Number of sample

	// Initial Median is the median of each channel in all matrix of dataset
	median = covs[0];				// to copy size
	for (size_t i = 0; i < median.size(); ++i)
	{
		std::vector<double> tmp;
		tmp.reserve(n);					// Reserve to optimize (a little) the pushback memory access.
		for (const auto& cov : covs) { tmp.push_back(cov.data()[i]); }	// Stack value number i of all matrix
		median.data()[i] = Median(tmp);
	}

	size_t iter = 0;					// number of iteration
	double gain = epsilon;				// Gain since last compute
	while (iter < maxIter && gain >= epsilon)
	{
		Eigen::MatrixXd prev = median;	// Keep old median
		median.setZero();				// Reset median
		double sumCoefs = 0;			// Sum of Coefficient
		std::vector<double> coefs;		// Coefficient for each sample
		coefs.reserve(n);				// Reserve to optimize (a little) the pushback memory access.
		for (const auto& cov : covs)
		{
			Eigen::MatrixXd difference = cov - prev;
			double coef = 1.0 / sqrt(difference.cwiseProduct(difference).sum());
			coefs.push_back(coef);
			sumCoefs += coef;			// Sum for normalization
			median += coef * cov;		// Add to the new median
		}
		median /= sumCoefs;				// Normalize
		
		gain = (median - prev).norm() / median.norm();	// It's the Frobenius norm
		iter++;
	}
	return true;
}
//---------------------------------------------------------------------------------------------------