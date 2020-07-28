///-------------------------------------------------------------------------------------------------
/// 
/// \file Mean.hpp
/// \brief All functions to estimate the mean of Vector of Covariance Matrix.
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 26/10/2018.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// \remarks 
/// - List of Metrics inspired by the work of Alexandre Barachant : <a href="https://github.com/alexandrebarachant/pyRiemann">pyRiemann</a> (<a href="https://github.com/alexandrebarachant/pyRiemann/blob/master/LICENSE">License</a>).
/// - The Approximate joint diagonalization based on pham's algorithm is not implemented.
/// - The Approximate joint diagonalization based log-Euclidean (ALE) Mean doesn't work => Need to implement <see cref="AJDPham"/> and check if it works next.
/// - The Wasserstein Mean Doesn't work so good (after \f$10^{-3}\f$ precision with the pyriemann library).
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>
#include <vector>

/// <summary>	Compute the median of vector of covariance matrix. </summary>
/// <param name="covs">  	Vector of Covariance Matrix. </param>
/// <param name="median">  	The computed median. </param>
/// <param name="epsilon"> 	(Optional) The epsilon value to stop algorithm. </param>
/// <param name="maxIter">	(Optional) The maximum iteration allowed to find best Median. </param>
/// <returns>	True if it succeeds, false if it fails. </returns>
/// <remarks>  it's an iteratively algorithm, so we have a limit of iteration and an epsilon value to consider the calculation as satisfactory. </remarks>
bool Median(const std::vector<Eigen::MatrixXd>& covs, Eigen::MatrixXd& median, double epsilon = 0.0001, int maxIter = 50);
