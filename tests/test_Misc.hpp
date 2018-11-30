///-------------------------------------------------------------------------------------------------
/// 
/// \file test_RiemannUtils.hpp
/// 
/// \brief Some constants and functions for google tests
/// 
/// \author Thibaut Monseigne (Inria).
/// 
/// \version 0.1.
/// 
/// \date 26/10/2018.
/// 
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// 
/// \remarks 
/// - For this test I compare the results with the <a href="https://github.com/alexandrebarachant/pyRiemann">pyRiemann</a> library (<a href="https://github.com/alexandrebarachant/pyRiemann/blob/master/LICENSE">License</a>) or <a href="http://scikit-learn.org">sklearn</a> if pyRiemman just redirect the function.
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <type_traits>
#include "utils/Covariance.hpp"
#include "utils/Featurization.hpp"

#define REPEAT_TEST 2

Eigen::MatrixXd M1, M2, M1Cov, M2Cov;
Eigen::RowVectorXd V1, M1Tan, M2Tan;
std::vector<Eigen::MatrixXd> VCov;
const std::string Sep = "\n====================\n";

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Initializes The 6 Main Objects. </summary>
///
///-------------------------------------------------------------------------------------------------
inline void Initialize()
{
	M1.resize(2, 10);
	M1 << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
		20, 19, 18, 17, 16, 15, 14, 13, 12, 11;

	M2.resize(2, 10);
	M2 << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
		1, 2, 3, 4, 5, 6, 7, 8, 9, 10;

	V1.resize(10);
	V1 << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;

	CovarianceMatrix(M1, M1Cov, Estimator_LWF, true);	// LedoitWolf assure SPD Matrix
	CovarianceMatrix(M2, M2Cov, Estimator_LWF, true);	// LedoitWolf assure SPD Matrix

	VCov.resize(2);
	VCov[0] = M1Cov;
	VCov[1] = M2Cov;
	const Eigen::MatrixXd I_N = Eigen::MatrixXd::Identity(2, 2);
	TangentSpace(M1Cov, M1Tan, I_N);
	TangentSpace(M2Cov, M2Tan, I_N);
}

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Check if two doubles are almost equal. </summary>
///
/// <param name="x">	  	The first double. </param>
/// <param name="y">	  	The second double. </param>
/// <param name="epsilon">	(Optional) The epsilon. </param>
///
/// <returns>	True if almost equal, false if not. </returns>
/// 
///-------------------------------------------------------------------------------------------------
inline bool isAlmostEqual(const double x, const double y, const double epsilon = 0.0001)
{
	return abs(x - y) < epsilon;
}

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Check if sum of two vectors are almost equal. </summary>
///
/// <typeparam name="T">	Generic numeric type parameter. </typeparam>
///
/// <param name="x">	The first vector of numeric values. </param>
/// <param name="y">	The second vector of numeric values. </param>
/// <param name="epsilon">	(Optional) The epsilon. </param>
///
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
///-------------------------------------------------------------------------------------------------
template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
bool isAlmostEqual(const std::vector<T>& x, const std::vector<T>& y, const double epsilon = 0.0001)
{
	double xsum = 0.0, ysum = 0.0;
	for (const auto& n : x) { xsum += n; }
	for (const auto& n : y) { ysum += n; }
	return isAlmostEqual(xsum, ysum, epsilon);
}

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Check if sum of two matrix are almost equal. </summary>
///
/// <typeparam name="T">	Generic numeric type parameter. </typeparam>
///
/// <param name="x">	The first matrix. </param>
/// <param name="y">	The second matrix. </param>
/// <param name="epsilon">	(Optional) The epsilon. </param>
///
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
///-------------------------------------------------------------------------------------------------
inline bool isAlmostEqual(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const double epsilon = 0.0001)
{
	return isAlmostEqual(x.cwiseAbs().sum(), y.cwiseAbs().sum(), epsilon);
}

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Error message for doubles. </summary>
///
/// <param name="name">	The name of the test. </param>
/// <param name="ref"> 	The reference value. </param>
/// <param name="calc">	The calculate value. </param>
///
/// <returns>	Error message. </returns>
/// 
///-------------------------------------------------------------------------------------------------
inline std::stringstream ErrorMsg(const std::string& name, const double ref, const double calc)
{
	std::stringstream  ss;
	ss << Sep << name << " : " << std::endl << "  Ref : \t" << ref << std::endl << "  Calc : \t" << calc << Sep;
	return ss;
}

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Error message for numeric vector. </summary>
///
/// <typeparam name="T">	Generic numeric type parameter. </typeparam>
/// 
/// <param name="name">	The name of the test. </param>
/// <param name="ref"> 	The reference value. </param>
/// <param name="calc">	The calculate value. </param>
///
/// <returns>	Error message. </returns>
/// 
///-------------------------------------------------------------------------------------------------
template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
inline std::stringstream ErrorMsg(const std::string& name, const std::vector<T>& ref, const std::vector<T>& calc)
{
	std::stringstream  ss;
	ss << Sep << name << " : " << std::endl << "  Ref : \t[";
	for (const T& t : ref) {
		ss << t << ", ";
	}
	ss << "\b\b]" << std::endl << "  Calc : \t[";
	for (const T& t : calc) {
		ss << t << ", ";
	}
	ss << "\b\b]" << Sep;
	return ss;
}

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Error message for matrix. </summary>
///
/// <param name="name">	The name of the test. </param>
/// <param name="ref"> 	The reference value. </param>
/// <param name="calc">	The calculate value. </param>
///
/// <returns>	Error message. </returns>
/// 
///-------------------------------------------------------------------------------------------------
inline std::stringstream ErrorMsg(const std::string& name, const Eigen::MatrixXd& ref, const Eigen::MatrixXd& calc)
{
	std::stringstream  ss;
	ss << Sep << name << " : " << std::endl << "********** Ref **********\n" << ref << std::endl << "********** Calc **********\n" << calc << Sep;
	return ss;
}