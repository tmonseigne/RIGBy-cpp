///-------------------------------------------------------------------------------------------------
/// 
/// \file Misc.hpp
/// \brief All misc functions (Matrix Median, Riemannian Eigen Values).
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 29/07/2020.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// \remarks This algortihms is inspired by the plugin clean_rawdata in <a href="https://sccn.ucsd.edu/eeglab/index.php">EEGLAB</a> (<a href="https://github.com/sccn/clean_rawdata/blob/master/LICENSE">License</a>).
///
///-------------------------------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>
#include <vector>
#include "geometry/Metrics.hpp"

namespace Geometry {

//---------------------------------------------------------------------------
//------------------------------ Matrix Median ------------------------------
//---------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
/// <summary> Find the median of stl vector. </summary>
/// <typeparam name="T"> The type of the values (only arithmetic type). </typeparam>
/// <param name="v"> the vector of values. </param>
/// <returns> The median of vector. </returns>
template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
T Median(const std::vector<T>& v)
{
	std::vector<T> tmp = v;
	const size_t n = tmp.size() / 2;									// Where is the middle (if odd number of value the decimal part is floor by cast)
	std::nth_element(tmp.begin(), tmp.begin() + n, tmp.end());			// We sort only until the limit usefull
	return (tmp.size() % 2 == 0) ? (tmp[n] + tmp[n - 1]) / 2 : tmp[n];	// For Even number of value we take the mean of the two middle value
}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
/// <summary> Find the median of values of the Eigen Matrix. </summary>
/// <param name="m"> the matrix. </param>
/// <returns> The median of matrix. </returns>
double Median(const Eigen::MatrixXd& m);
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
/// <summary>	Compute the median of vector of matrix with the Weiszfeld's algorithm. <br/> 
/// To compute this median, we start by computing the initial median of the dataset by taking each element of the matrices independently.
/// That is to say that for the element at position i, j(a_i, j) of the matrices, we computes the median of the elements a_i, j of all the matrices of the dataset.
/// We thus have an initial median for our dataset. <br/> 
/// Then, we refine our median by the iterative algorithm of Weiszfeld:  
/// - We remove the median in our dataset.
/// - For each new matrices, we compute the norm.
/// - We sum the the matrices in initial dataset (divided by their own norm) and we normalize the result by the sum of inverse norms.
/// - We iterate this previous step until we have a difference between the old and new median is under an epsilon or that the number of iterations is above the limit.
/// </summary>
/// <param name="matrices">	Vector of Matrix. </param>
/// <param name="median">	The computed median. </param>
/// <param name="epsilon">	(Optional) The epsilon value to stop algorithm. </param>
/// <param name="maxIter">	(Optional) The maximum iteration allowed to find best Median. </param>
/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
/// <remarks>  it's an iteratively algorithm, so we have a limit of iteration and an epsilon value to consider the calculation as satisfactory. </remarks>
bool MedianEuclidian(const std::vector<Eigen::MatrixXd>& matrices, Eigen::MatrixXd& median, const double epsilon = 0.0001, const size_t maxIter = 50);
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
/// <summary>	Compute the median of vector of matrix with the Riemman Barycentre. <br/>
/// - Initialize the median with the euclidian mean of matrices.
/// - Iterate until the stop criterion (<c>iteration</c> over <c>maxIter</c> or \f$\text{gain}\f$ under <c>epsilon</c>).
///   - Compute the tangent space projection of each matrices with median as reference.
///   - Compute the sum (\f$\mathcal{S}\f$) of euclidian distance of each tangent space projection. <br/>
///   \f[ \delta_E=\sqrt{\sum_{i \in N}{x_i^2}} \quad \text{with } x_i \text{ the feature } i \text{ of the tangent space projection}\f]
///   - Compare with previous sum and stop if \f$\text{gain} < \varepsilon\f$. <br/>
///   \f[ \text{gain} = \left|\frac{\mathcal{S} - \mathcal{S}_\text{prev}}{\mathcal{S}_\text{prev}}\right| \f]
///   - Compute Median of each feature \f$i\f$ of tangent space projection.
///   - Transform this tangent space projection median to riemann space with previous median as reference and update the median by this new matrix.
/// </summary>
/// <param name="matrices">	Vector of Matrix. </param>
/// <param name="median">	The computed median. </param>
/// <param name="epsilon">	(Optional) The epsilon value to stop algorithm. </param>
/// <param name="maxIter">	(Optional) The maximum iteration allowed to find best Median. </param>
/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
bool MedianRiemann(const std::vector<Eigen::MatrixXd>& matrices, Eigen::MatrixXd& median, const double epsilon = 0.0001, const size_t maxIter = 50);
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
/// <summary> Give the identity matrix has median. </summary>
/// <param name="matrices">	Vector of Matrix. </param>
/// <param name="median">	The computed median. </param>
/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
bool MedianIdentity(const std::vector<Eigen::MatrixXd>& matrices, Eigen::MatrixXd& median);
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
/// <summary>	Compute the median of vector of matrix with the Weiszfeld's algorithm for Euclidian Metric and Riemman Barycentre. </summary>
/// <param name="matrices">	Vector of Matrix. </param>
/// <param name="median">	The computed median. </param>
/// <param name="epsilon">	(Optional) The epsilon value to stop algorithm. </param>
/// <param name="maxIter">	(Optional) The maximum iteration allowed to find best Median. </param>
/// <param name="metric">	(Optional) THe metric to use. </param>
/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
/// <remarks>  it's an iteratively algorithm, so we have a limit of iteration and an epsilon value to consider the calculation as satisfactory. </remarks>
bool Median(const std::vector<Eigen::MatrixXd>& matrices, Eigen::MatrixXd& median, const double epsilon = 0.0001, const size_t maxIter = 50, const EMetric& metric = EMetric::Euclidian);
//-------------------------------------------------------------------------------------------------

}  // namespace Geometry
