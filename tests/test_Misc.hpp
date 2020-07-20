///-------------------------------------------------------------------------------------------------
/// 
/// \file test_Misc.hpp
/// \brief Some constants and functions for google tests
/// \author Thibaut Monseigne (Inria).
/// \version 0.1.
/// \date 26/10/2018.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// \remarks 
/// - For this test I compare the results with the <a href="https://github.com/alexandrebarachant/pyRiemann">pyRiemann</a> library (<a href="https://github.com/alexandrebarachant/pyRiemann/blob/master/LICENSE">License</a>) or <a href="http://scikit-learn.org">sklearn</a> if pyRiemman just redirect the function.
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include <cmath>
#include <vector>
#include <type_traits>

#include "classifier/IMatrixClassifier.hpp"

const std::string SEP = "\n====================\n";

//*********************************************************************************
//********** Comparison of values with epsilon tolerance for google test **********
//*********************************************************************************
/// <summary>	Check if two doubles are almost equal. </summary>
/// <param name="x">	  	The first value. </param>
/// <param name="y">	  	The second value. </param>
/// <param name="epsilon">	(Optional) The epsilon tolerance. </param>
/// <returns>	True if almost equal, false if not. </returns>
inline bool isAlmostEqual(const double x, const double y, const double epsilon = 0.0001) { return std::abs(x - y) < epsilon; }

/// <summary>	Check if sum of two vectors are almost equal. </summary>
/// <typeparam name="T">	Generic numeric type parameter. </typeparam>
/// \copydetails isAlmostEqual(const double, const double, const double)
template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
bool isAlmostEqual(const std::vector<T>& x, const std::vector<T>& y, const double epsilon = 0.0001)
{
	double xsum = 0.0, ysum = 0.0;
	for (const auto& n : x) { xsum += n; }
	for (const auto& n : y) { ysum += n; }
	return (x.size() == y.size() && isAlmostEqual(xsum, ysum, epsilon));
}

/// <summary>	Check if sum of two matrix are almost equal. </summary>
/// \copydetails isAlmostEqual(const double, const double, const double)
inline bool isAlmostEqual(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const double epsilon = 0.0001)
{
	return x.size() == y.size() && isAlmostEqual(x.cwiseAbs().sum(), y.cwiseAbs().sum(), epsilon);
}

//*****************************************************************
//********** Error Message Standardization for googltest **********
//*****************************************************************
/// <summary>	Error message for size_t. </summary>
/// <param name="name">	The name of the test. </param>
/// <param name="ref"> 	The reference value. </param>
/// <param name="calc">	The calculate value. </param>
/// <returns>	Error message. </returns>
inline std::stringstream ErrorMsg(const std::string& name, const size_t ref, const size_t calc)
{
	std::stringstream ss;
	ss << SEP << name << " : Reference : " << ref << ", \tCompute : " << calc << SEP;
	return ss;
}

/// <summary>	Error message for doubles. </summary>
/// \copydetails ErrorMsg(const std::string&, const size_t, const size_t)
inline std::stringstream ErrorMsg(const std::string& name, const double ref, const double calc)
{
	std::stringstream ss;
	ss << SEP << name << " : Reference : " << ref << ", \tCompute : " << calc << SEP;
	return ss;
}

/// <summary>	Error message for numeric vector. </summary>
/// <typeparam name="T">	Generic numeric type parameter. </typeparam>
/// \copydetails ErrorMsg(const std::string&, const size_t, const size_t)
template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
std::stringstream ErrorMsg(const std::string& name, const std::vector<T>& ref, const std::vector<T>& calc)
{
	std::stringstream ss;
	ss << SEP << name << " : " << std::endl << "  Reference : \t[";
	for (const T& t : ref) { ss << t << ", "; }
	if (!ref.empty()) { ss.seekp(ss.str().length() - 2); }
	ss << "] " << std::endl << "  Compute : \t[";
	for (const T& t : calc) { ss << t << ", "; }
	if (!ref.empty()) { ss.seekp(ss.str().length() - 2); }
	ss << "] " << SEP;
	return ss;
}

/// <summary>	Error message for matrix. </summary>
/// \copydetails ErrorMsg(const std::string&, const size_t, const size_t)
inline std::stringstream ErrorMsg(const std::string& name, const Eigen::MatrixXd& ref, const Eigen::MatrixXd& calc)
{
	std::stringstream ss;
	ss << SEP << name << " : " << std::endl << "********** Reference **********\n" << ref << std::endl << "********** Compute **********\n" << calc << SEP;
	return ss;
}

/// <summary>	Error message for matrix Classifier. </summary>
/// \copydetails ErrorMsg(const std::string&, const size_t, const size_t)
inline std::stringstream ErrorMsg(const std::string& name, const IMatrixClassifier& ref, const IMatrixClassifier& calc)
{
	std::stringstream ss;
	ss << SEP << name << " : " << std::endl << "********** Reference **********\n" << ref << std::endl << "********** Compute **********\n" << calc << SEP;
	return ss;
}
