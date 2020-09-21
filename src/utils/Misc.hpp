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

//---------------------------------------------------------------------------
//------------------------------ Matrix Median ------------------------------
//---------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
/// <summary> Find the median of stl vector. </summary>
/// <typeparam name="T"> The type of the values (only arithmetic type). </typeparam>
/// <param name="v"> the vector of values. </param>
/// <returns> The median of matrix. </returns>
template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
T Median(std::vector<T>& v)
{
	const size_t n = v.size() / 2;								// Where is the middle (if odd number of value the decimal part is floor by cast)
	std::nth_element(v.begin(), v.begin() + n + 1, v.end());	// We sort one number more if we have an even number of value
	return (v.size() % 2 == 0) ? (v[n] + v[n + 1]) / 2 : v[n];	// For Even number of value we take the mean of the two middle value
}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
/// <summary> Find the median of values of the Eigen Matrix. </summary>
/// <param name="m"> the matrix. </param>
/// <returns> The median of matrix. </returns>
double Median(const Eigen::MatrixXd& m);
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
/// <summary>	Compute the median of vector of matrix with the Weiszfeld's algorithm.\n
///
/// To compute this median, we start by computing the initial median of the dataset by taking each element of the matrices independently.
/// That is to say that for the element at position i, j(a_i, j) of the matrices, we computes the median of the elements a_i, j of all the matrices of the dataset.
/// We thus have an initial median for our dataset.\n
///
/// Then, we refine our median by the iterative algorithm of Weiszfeld:  
/// - We remove the median in our dataset.
/// - For each new matrices, we compute the norm.
/// - We sum the the matrices in initial dataset (divided by their own norm) and we normalize the result by the sum of inverse norms.
/// - We iterate this previous step until we have a difference between the old and new median is under an epsilon or that the number of iterations is above the limit.
/// </summary>
/// <param name="matrices">  	Vector of Matrix. </param>
/// <param name="median">  	The computed median. </param>
/// <param name="epsilon"> 	(Optional) The epsilon value to stop algorithm. </param>
/// <param name="maxIter">	(Optional) The maximum iteration allowed to find best Median. </param>
/// <returns>	True if it succeeds, false if it fails. </returns>
/// <remarks>  it's an iteratively algorithm, so we have a limit of iteration and an epsilon value to consider the calculation as satisfactory. </remarks>
bool Median(const std::vector<Eigen::MatrixXd>& matrices, Eigen::MatrixXd& median, const double epsilon = 0.0001, const int maxIter = 50);
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------
//------------------------------ Range ------------------------------
//-------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
/// <summary>	Create a range of double value. </summary>
/// <param name="begin">	Begin of the range. </param>
/// <param name="end">		End of the range. </param>
/// <param name="step">		Step of the range. </param>
/// <param name="closed">	Autorize the end in range if <c>true</c>. </param>
/// <returns> The range vector. </returns>
/// <remarks>Use [std::iota](https://en.cppreference.com/w/cpp/algorithm/iota) function and a struct for this specific used. </remarks>
std::vector<double> doubleRange(const double begin, const double end, const double step = 1.0, const bool closed = true);
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
/// <summary>	Create a range of index with double value rounded. </summary>
/// <param name="begin">	Begin of the range. </param>
/// <param name="end">		End of the range. </param>
/// <param name="step">		Step of the range. </param>
/// <param name="closed">	Autorize the end in range if <c>true</c>. </param>
/// <param name="unique">	Remove duplicate value if <c>true</c>. </param>
/// <returns> The range vector. </returns>
/// <remarks>Use [std::iota](https://en.cppreference.com/w/cpp/algorithm/iota) function and a struct for this specific used. </remarks>
std::vector<size_t> RoundIndexRange(const double begin, const double end, const double step, const bool closed = true, const bool unique = true);
//-------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------ Fit Distribution ------------------------------
//------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
/// <summary>	Commpute histogramm of dataset extended in n bins. </summary>
/// <param name="dataset">	Input vector (all datas are positive). </param>
/// <param name="n">		Number of bin of the final histogram. </param>
std::vector<size_t> BinHist(const std::vector<double>& dataset, const size_t n);
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
/// <summary> Get a Fit distribution. </summary>
/// <param name="values">		The values.</param>
/// <param name="mu">			The mu.</param>
/// <param name="sigma">		The sigma.</param>
/// <param name="betas">		</param>
/// <param name="minQuant">		</param>
/// <param name="maxQuant">		</param>
/// <param name="minClean">		</param>
/// <param name="maxDropout">	</param>
/// <param name="stepBound">	</param>
/// <param name="stepScale">	</param>
/// <returns></returns>
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
/// <summary>	Compute the eigen vector of the input matrix. </summary>
/// <param name="matrix">  			input Matrix. </param>
/// <param name="eigenVector">  	Eigen Vector of input input matrix. </param>
/// <returns>	True if it succeeds, false if it fails. </returns>
/// <remarks> This algorithm is in <a href="https://sccn.ucsd.edu/eeglab/index.php">EEGLAB</a> plugin and inspired by the paper "A Riemannian Newton Algorithm for Nonlinear Eigenvalue Problems", Zhi Zhao, Zheng - Jian Bai, and Xiao - Qing Jin, SIAM Journal on Matrix Analysisand Applications, 36(2), 752 - 774, 2015. </remarks>
bool RiemannianNonLinearEigenVector(const Eigen::MatrixXd& matrix, Eigen::MatrixXd& eigenVector);
//-------------------------------------------------------------------------------------------------
