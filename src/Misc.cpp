#include "geometry/Misc.hpp"
#include "geometry/Featurization.hpp"

#include <boost/math/special_functions/gamma.hpp>
#include <numeric>	// std::iota

namespace Geometry {

///-------------------------------------------------------------------------------------------------
/// <summary>	Get the sign of the specified value. </summary>
/// <param name="x">	The value. </param>
/// <returns>	<c>1</c> if <c>x > 0</c>, <c>0</c> if <c>x == 0</c>, <c>-1</c> if <c>x < 0</c>. </returns>
template <typename T>
int sgn(T x) { return (T(0) < x) - (x < T(0)); }
///-------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
/// <summary> Structure used for iota function for range of double. </summary>
struct SDoubleIota
{
	explicit SDoubleIota(const double init = 0.0, const double inc = 1.0) : v(init), inc(inc) {}

	operator double() const { return v; }	// don't add explicit qualifier for iota functions (were template cast is used)
	SDoubleIota& operator++()
	{
		v += inc;
		return *this;
	}
	double v;
	double inc;
};

//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
/// <summary> Structure used for iota function for range of round index. </summary>
struct SRoundIndex
{
	explicit SRoundIndex(const double init = 0.0, const double inc = 1.0) : v(init), inc(inc) {}

	operator size_t() const { return size_t(std::round(v)); }	// don't add explicit qualifier for iota functions (were template cast is used)
	SRoundIndex& operator++()
	{
		v += inc;
		return *this;
	}
	double v;
	double inc;
};

//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
std::vector<double> doubleRange(const double begin, const double end, const double step, const bool closed)
{
	std::vector<double> res;
	if (end < begin) { return res; }
	const double size = (end - begin) / step;
	// check if the end is inclued in range for size, we ceil for no modulo values and add the last value to the range 
	res.resize(size_t(std::ceil((closed && std::trunc(size) == size) ? size + 1 : size)));
	std::iota(res.begin(), res.end(), SDoubleIota(begin, step));

	return res;
}

//---------------------------------------------------------------------------------------------------
std::vector<size_t> RoundIndexRange(const double begin, const double end, const double step, const bool closed, const bool unique)
{
	std::vector<size_t> res;
	if (end < begin) { return res; }
	const double size = (end - begin) / step;
	// check if the end is inclued in range for size, we ceil for no modulo values and add the last value to the range 
	res.resize(size_t(std::ceil((closed && std::trunc(size) == size) ? size + 1 : size)));
	std::iota(res.begin(), res.end(), SRoundIndex(begin, step));

	if (unique)
	{
		const auto last = std::unique(res.begin(), res.end());	// Remove duplicate values (but after last, we have undefined value)
		res.erase(last, res.end());								// Resize Vector (we erase after last)
	}
	return res;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
std::vector<size_t> BinHist(const std::vector<double>& dataset, const size_t n)
{
	std::vector<size_t> res(n, 0);
	const double max = *std::max_element(dataset.begin(), dataset.end());
	if (max == 0) { return res; }				// if max is 0, coef can't be compute
	const double coef = n / max;
	for (const auto& data : dataset)
	{
		const size_t bin = size_t(std::floor(data * coef));
		if (bin < n) { res[bin]++; }
		else if (bin == n) { res[n - 1]++; }	// if this data is equal to max
	}
	return res;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool FitDistribution(const std::vector<double>& values, double& mu, double& sigma, const std::vector<double>& betas, const double minQuant,
					 const double maxQuant, const double minClean, const double maxDropout, const double stepBound, const double stepScale)
{
	if (values.empty() || betas.empty() || minQuant < 0 || minQuant > 1 || maxQuant < 0 || maxQuant > 1 || minClean < 0 || maxDropout < 0
		|| stepBound < 0.0001 || stepBound > 0.1 || stepScale < 0.0001 || stepScale > 0.1) { return false; }

	//========== Scales ==========
	const size_t nBeta = betas.size();
	// Scales is a vector for each beta as :
	// scale = beta/(2*gamma(1/beta)) with gamma the function as gamma(n) = (n-1)! for all integer greater than 0
	std::vector<double> scales;
	scales.reserve(nBeta);
	std::transform(betas.begin(), betas.end(), std::back_inserter(scales), [](const double beta) -> double { return beta / (2 * tgamma(1 / beta)); });

	//========== zBounds ==========
	// zBounds is a vector of lower and upper bounds for each beta as : sign(quants-1/2) * gammaincinv(sign(quants-1/2) * (2*quants-1), 1/beta)^(1/beta);
	// with gammaincinv the Inverse incomplete gamma function, here quants are the quantiles limit (by default [0.022 0.6])
	std::vector<std::vector<double>> zBounds(nBeta);
	const int signMin    = sgn(minQuant - 0.5), signMax          = sgn(maxQuant - 0.5);
	const double coefMin = signMin * (2 * minQuant - 1), coefMax = signMax * (2 * maxQuant - 1);

	for (size_t i = 0; i < nBeta; ++i)
	{
		if (betas[i] == 0) { zBounds[i] = { 0, 0 }; }
		else
		{
			const double beta = 1 / betas[i];
			zBounds[i]        = {
				signMin * pow(boost::math::gamma_p_inv(beta, coefMin), beta),
				signMax * pow(boost::math::gamma_p_inv(beta, coefMax), beta)
			};
		}
	}

	//========== Sort Values ==========
	// We sort values to access quantiles directly
	const size_t n                = values.size();
	std::vector<double> newValues = values;
	std::sort(newValues.begin(), newValues.end());

	//========== Compute Index range ==========
	// Width are the limit if all data is clean or artifacted. It's usefull for the for loop limit and step for each width possible
	// Bounds are the range of begining value used to compute mu and sigma. It's usefull for the for loop limit and step for first index of value to take
	// We create Vector for widths and bounds to precompute all round and avoid duplicate indexes in widths or bounds
	std::vector<size_t> widths = RoundIndexRange(n * (maxQuant - minQuant) * minClean, n * (maxQuant - minQuant), n * stepScale, true, false);
	std::reverse(widths.begin(), widths.end());
	const std::vector<size_t> bounds = RoundIndexRange(n * minQuant, n * (minQuant + maxDropout), n * stepBound, true, false);
	const size_t maxWidth            = std::max(widths.front(), widths.back());	// to prevent if widths is in descending or ascending order
	const size_t nBound              = bounds.size();

	//========== Compute Grid (with index range) ==========
	// Create the Biggest table of data with width in column and bound in row
	std::vector<std::vector<double>> grid(nBound);
	std::vector<double> firsts(nBound);
	for (size_t i = 0; i < nBound; ++i)
	{
		grid[i].reserve(maxWidth);
		const auto first = newValues.begin() + bounds[i];
		std::copy_n(first, maxWidth, std::back_inserter(grid[i]));
		firsts[i] = grid[i][0];
		for (auto& e : grid[i]) { e -= firsts[i]; }	// Substract first value on all element
	}

	//==========  Width Loop ==========
	double bestKl   = std::numeric_limits<double>::max();
	size_t bestBeta = 0, bestId = 0, bestWidth = 0;
	// for each interval width...
	for (const auto& w : widths)
	{
		const size_t nbins = size_t(std::round(3 * log2(1 + (double(w) / 2))));

		//==========  Compute Histogramm ==========
		std::vector<std::vector<double>> hist(nBound);
		for (size_t i = 0; i < nBound; ++i)
		{
			hist[i].reserve(nbins);
			std::vector<size_t> tmp = BinHist(std::vector<double>(grid[i].begin(), grid[i].begin() + w), nbins);
			std::transform(tmp.begin(), tmp.end(), std::back_inserter(hist[i]), [](const size_t e) -> double { return log(e + 0.01); });
		}

		//==========  Beta Loop ==========
		for (size_t b = 0; b < nBeta; ++b)
		{
			//==========  Compute Probability ==========
			std::vector<double> prob(nbins);
			double sumprob = 0.0;
			for (size_t i = 0; i < nbins; ++i)
			{
				prob[i] = std::exp(-std::pow(std::abs(zBounds[b][0] + (((i + 0.5) / nbins) * (zBounds[b][1] - zBounds[b][0]))), betas[b])) * scales[b];
				sumprob += prob[i];
			}
			if (sumprob != 0) { for (auto& p : prob) { p /= sumprob; } }

			//========== Compute the Kullback-Leibler divergences ==========
			//kl = sum(prob * (log(prob) - hist)) + log(w));
			std::vector<double> kl(nBound, log(w));
			for (size_t i = 0; i < nBound; ++i) { for (size_t j = 0; j < nbins; ++j) { kl[i] += prob[j] * (log(prob[j]) - hist[i][j]); } }

			// Update Parameters
			auto minIt = std::min_element(kl.begin(), kl.end());
			if (*minIt < bestKl)
			{
				bestKl    = *minIt;
				bestBeta  = b;
				bestId    = minIt - kl.begin();
				bestWidth = w - 1;
			}
		}
	}

	double alpha = grid[bestId][bestWidth] / (zBounds[bestBeta][1] - zBounds[bestBeta][0]);
	double beta  = betas[bestBeta];

	mu    = firsts[bestId] - zBounds[bestBeta][0] * alpha;
	sigma = sqrt(alpha * alpha * std::tgamma(3 / beta) / std::tgamma(1 / beta));

	return true;
}
//---------------------------------------------------------------------------------------------------
void sortedEigenVector(const Eigen::MatrixXd& matrix, Eigen::MatrixXd& vectors, std::vector<double>& values, const EMetric /*metric*/)
{
	// Compute Eigen Vector/Values
	const Eigen::EigenSolver<Eigen::MatrixXd> es(matrix);
	const Eigen::MatrixXd tmpVec = es.eigenvectors().real();	// It's complex by default but all imaginary part are 0
	const Eigen::MatrixXd tmpVal = es.eigenvalues().real();		// It's complex by default but all imaginary part are 0
	values                       = std::vector<double>(tmpVal.data(), tmpVal.data() + tmpVal.size());

	// Get order of eigen values.
	std::vector<size_t> idx(values.size());
	std::iota(idx.begin(), idx.end(), 0);
	std::stable_sort(idx.begin(), idx.end(), [&values](const size_t i1, const size_t i2) { return values[i1] < values[i2]; });
	// Sort Eigen Values
	std::stable_sort(values.begin(), values.end());

	// Sort Eigen Vector
	vectors = tmpVec;	// copy matrix to set size easily
	for (size_t i = 0; i < size_t(tmpVec.cols()); ++i) { vectors.col(i) = tmpVec.col(idx[i]); }
}
//---------------------------------------------------------------------------------------------------

}  // namespace Geometry
