///-------------------------------------------------------------------------------------------------
/// 
/// \file Median.hpp
/// \brief All functions to estimate the median of Vector of Covariance Matrix.
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 29/07/2020.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// \remarks This algortihm is inspired by the plugin clean_rawdata in <a href="https://sccn.ucsd.edu/eeglab/index.php">EEGLAB</a> (<a href="https://github.com/sccn/clean_rawdata/blob/master/LICENSE">License</a>).
///
///-------------------------------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>
#include <vector>

/// <summary> Find the median of stl vector. </summary>
/// <typeparam name="T"> The type of the values (only arithmetic type). </typeparam>
/// <param name="v"> the vector of values. </param>
/// <returns> The median of matrix. </returns>
template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
T Median(std::vector<T>& v)
{
	const size_t n = v.size() / 2;								// Where is the middle (if odd number of value the decimal part is floor by cast)
	std::nth_element(v.begin(), v.begin() + n, v.end());		// We sort one number more if we have an even number of value
	return (v.size() % 2 == 0) ? (v[n] + v[n - 1]) / 2 : v[n];	// For Even number of value we take the mean of the two middle value
}

/// <summary> Find the median of values of the Eigen Matrix. </summary>
/// <param name="m"> the matrix. </param>
/// <returns> The median of matrix. </returns>
double Median(const Eigen::MatrixXd& m);

/// <summary>	Compute the median of vector of covariance matrix with the Weiszfeld's algorithm. </summary>
/// <param name="covs">  	Vector of Covariance Matrix. </param>
/// <param name="median">  	The computed median. </param>
/// <param name="epsilon"> 	(Optional) The epsilon value to stop algorithm. </param>
/// <param name="maxIter">	(Optional) The maximum iteration allowed to find best Median. </param>
/// <returns>	True if it succeeds, false if it fails. </returns>
/// <remarks>  it's an iteratively algorithm, so we have a limit of iteration and an epsilon value to consider the calculation as satisfactory. </remarks>
bool Median(const std::vector<Eigen::MatrixXd>& covs, Eigen::MatrixXd& median, const double epsilon = 0.0001, const int maxIter = 50);
