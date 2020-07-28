#include "Median.hpp"

using namespace Eigen;
using namespace std;

//static const double EPSILON = 0.000000001;	// 10^{-9}
static const double EPSILON  = 0.0001;			// 10^{-4}
static const size_t ITER_MAX = 50;


//---------------------------------------------------------------------------------------------------
/// <summary> Find the median of stl vector. </summary>
/// <param name="v"> the matrix. </param>
/// <returns> The median of matrix. </returns>
double Median(std::vector<double>& v)
{
	const size_t n = v.size() / 2;
	std::nth_element(v.begin(), v.begin() + n, v.end());
	return v[n];
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
/// <summary> Find the median of values of the Eigen Matrix. </summary>
/// <param name="m"> the matrix. </param>
/// <returns> The median of matrix. </returns>
double Median(const MatrixXd& m)
{
	std::vector<double> v(m.data(), m.data() + m.rows() * m.cols());
	return Median(v);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool Median(const std::vector<MatrixXd>& covs, MatrixXd& median, double epsilon, int maxIter)
{
	if (covs.empty()) { return false; }
	const size_t n = covs.size();
	// 1 Initial Median is the median of each value in all matrix of dataset
	// Median = [median(a_{0,0}) median(a_{0,1}) .... median(a_{0,n})]
	// 2 
	median = covs[0];
	for (size_t i = 0; i < median.size(); ++i)
	{
		std::vector<double> tmp;
		tmp.reserve(n);
		for (const auto& cov : covs) { tmp.push_back(cov.data()[i]); }
		median.data()[i] = Median(tmp);
	}

	size_t iter = 0;
	double gain = epsilon;
	while (iter < maxIter && gain >= epsilon)
	{
		MatrixXd prev = median;
		double sumCoefs = 0;		// Sum of Coefficient
		std::vector<double> coefs;	// Coefficient for each sample
		coefs.reserve(n);			// Reserve to optimize (a little) the pushback memory access.
		for (const auto& cov : covs)
		{
			MatrixXd difference = cov - median;
			double coef = 1.0 / sqrt(difference.cwiseProduct(difference).sum());
			coefs.push_back(coef);
			sumCoefs += coef;		// Sum for normalization
			median += coef * cov;	// Compute new median
		}
		median /= sumCoefs;			// Normalize
		
		gain = (median - prev).norm() / median.norm();
		iter++;
	}
	
	return true;
}
//---------------------------------------------------------------------------------------------------
