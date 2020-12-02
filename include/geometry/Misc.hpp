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

//-------------------------------------------------------------------
//------------------------------ Range ------------------------------
//-------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
/// <summary>	Create a range of double value. </summary>
/// <param name="begin">	Begin of the range. </param>
/// <param name="end">		End of the range. </param>
/// <param name="step">		Step of the range. </param>
/// <param name="closed">	Autorize the end in range if <c>True</c>. </param>
/// <returns> The range vector. </returns>
/// <remarks>Use [std::iota](https://en.cppreference.com/w/cpp/algorithm/iota) function and a struct for this specific used. </remarks>
std::vector<double> doubleRange(const double begin, const double end, const double step = 1.0, const bool closed = true);
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
/// <summary>	Create a range of index with double value rounded. </summary>
/// <param name="begin">	Begin of the range. </param>
/// <param name="end">		End of the range. </param>
/// <param name="step">		Step of the range. </param>
/// <param name="closed">	Autorize the end in range if <c>True</c>. </param>
/// <param name="unique">	Remove duplicate value if <c>True</c>. </param>
/// <returns> The range vector. </returns>
/// <remarks>Use [std::iota](https://en.cppreference.com/w/cpp/algorithm/iota) function and a struct for this specific used. </remarks>
std::vector<size_t> RoundIndexRange(const double begin, const double end, const double step, const bool closed = true, const bool unique = true);
//-------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------ Fit Distribution ------------------------------
//------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
/// <summary>	Commpute histogramm of dataset extended in <c>n</c> bins, bins are computes from \f$[0;max]\f$ (values) to \f$[0;n]\f$ (bins). </summary>
/// <param name="dataset">	Input vector (all datas are positive). </param>
/// <param name="n">		Number of bin of the final histogram. </param>
std::vector<size_t> BinHist(const std::vector<double>& dataset, const size_t n);
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
/// <summary> Get a Fit distribution. </summary>
/// <param name="values">		The values. </param>
/// <param name="mu">			The mu. </param>
/// <param name="sigma">		The sigma. </param>
/// <param name="betas">		List of wanted \f$\beta\f$ shapes. </param>
/// <param name="minQuant">		Minimum of wanted quantile. </param>
/// <param name="maxQuant">		Maximum of wanted quantile. </param>
/// <param name="minClean">		Minimum of estimated clean datas. </param>
/// <param name="maxDropout">	Maximum of estimated artifact datas. </param>
/// <param name="stepBound">	Step used to select beginning of datas subset. </param>
/// <param name="stepScale">	Step used to select size of datas subset. </param>
/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
bool FitDistribution(const std::vector<double>& values, double& mu, double& sigma,
					 const std::vector<double>& betas = doubleRange(1.7, 3.5, 0.15),
					 const double minQuant            = 0.022, const double maxQuant   = 0.60,
					 const double minClean            = 0.250, const double maxDropout = 0.10,
					 const double stepBound           = 0.010, const double stepScale  = 0.01);
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------
//------------------------------ Riemannian Eigen Values ------------------------------
//-------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
/// <summary>	Compute sorted eigen vector of the matrix. </summary>
/// <param name="matrix">	the input matrix. </param>
/// <param name="vectors"> Sorted eigen vectors. </param>
/// <param name="values">	Sorted eigen values. </param>
/// <param name="metric">	metric used for vectors. </param>
/// <remarks>	Actually only euclidian method is implemented. <br/> 
/// For Riemmanian metric, we must have some optimisation algorithm. </remarks>
void sortedEigenVector(const Eigen::MatrixXd& matrix, Eigen::MatrixXd& vectors, std::vector<double>& values, const EMetric metric = EMetric::Euclidian);
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
/// <summary>	Compute the eigen vector of the input matrix. </summary>
/// <param name="matrix">  			input Matrix. </param>
/// <param name="eigenVector">  	Eigen Vector of input input matrix. </param>
/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
/// <remarks> This algorithm is in <a href="https://sccn.ucsd.edu/eeglab/index.php">EEGLAB</a> plugin and inspired by the paper "A Riemannian Newton Algorithm for Nonlinear Eigenvalue Problems", Zhi Zhao, Zheng - Jian Bai, and Xiao - Qing Jin, SIAM Journal on Matrix Analysisand Applications, 36(2), 752 - 774, 2015. </remarks>
//bool RiemannianNonLinearEigenVector(const Eigen::MatrixXd& matrix, Eigen::MatrixXd& eigenVector);
//-------------------------------------------------------------------------------------------------

}  // namespace Geometry
