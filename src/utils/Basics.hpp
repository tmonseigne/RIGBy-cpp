///-------------------------------------------------------------------------------------------------
/// 
/// \file Basics.hpp
/// 
/// \brief Basic functions of Eigen matrix manipulation and verification.
/// 
/// \author Thibaut Monseigne (Inria).
/// 
/// \version 1.0.
/// 
/// \date 26/10/2018.
/// 
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>		// Ceil
#include <type_traits>	// Template type

//************************************************
//******************** Matrix ********************
//************************************************
///----------------------------------------------------------------------------------------------------
/// 
/// <summary>	Removes the mean of the vector to this one (destructive operation). </summary>
/// 
/// <param name="v">	The vector to center. </param>
/// 
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
///----------------------------------------------------------------------------------------------------
bool VectorCenter(Eigen::RowVectorXd& v);

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Removes the mean of the vector to this one (non destructive operation). </summary>
/// 
/// <param name="in"> 	The vector to center. </param>
/// <param name="out">	The vector centered. </param>
/// 
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
///-------------------------------------------------------------------------------------------------
bool VectorCenter(const Eigen::RowVectorXd& in, Eigen::RowVectorXd& out);

///----------------------------------------------------------------------------------------------------
/// 
/// <summary>	Removes the mean of each row at the matrix (destructive operation). </summary>
/// 
/// <param name="m">	The Matrix to center. </param>
/// 
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
///-------------------------------------------------------------------------------------------------
bool MatrixCenter(Eigen::MatrixXd& m);

///----------------------------------------------------------------------------------------------------
/// 
/// <summary>	Removes the mean of each row at the matrix (destructive operation). </summary>
/// 
/// <param name="in">	The Matrix to center. </param>
/// <param name="out">	The Matrix centered. </param>
/// 
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
///-------------------------------------------------------------------------------------------------
bool MatrixCenter(const Eigen::MatrixXd& in, Eigen::MatrixXd& out);


///----------------------------------------------------------------------------------------------------
///
/// <summary>	Give the string format of Matrix for OpenViBE Log. </summary>
///
/// <param name="m">	The Matrix to display. </param>
///
/// <returns>		The string format. </returns>
///
///----------------------------------------------------------------------------------------------------
std::string MatrixPrint(const Eigen::MatrixXd& m);
//************************************************
//************************************************
//************************************************

//*************************************************************
//******************** Index Manipulations ********************
//*************************************************************
///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Gets the items selected by the index. </summary>
/// 
/// <param name="row">		the original row. </param>
/// <param name="index">	Elements to select. </param>
/// 
/// <returns>	Row with selected elements. </returns>
/// 
///-------------------------------------------------------------------------------------------------
Eigen::RowVectorXd GetElements(const Eigen::RowVectorXd& row, const std::vector<size_t>& index);

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Numpy arange in C++. </summary>
/// 
/// <typeparam name="T">	Generic numeric type parameter. </typeparam>
/// 
/// <param name="start">	The start. </param>
/// <param name="stop"> 	The stop. </param>
/// <param name="step"> 	(Optional) Amount to increment by. </param>
/// 
/// <returns>	vector&lt;T&gt; </returns>
/// 
///-------------------------------------------------------------------------------------------------
template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
std::vector<T> ARange(const T start, const T stop, const T step = 1)
{
	std::vector<T> result;
	result.reserve(size_t(ceil(1.0 * (stop - start) / step)));
	for (T i = start; i < stop; i += step)
		result.push_back(i);
	return result;
}
//*************************************************************
//*************************************************************
//*************************************************************

//***************************************************
//******************** Validates ********************
//***************************************************
///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Validate if value is in [min;max]. </summary>
///
/// <param name="value">	The value. </param>
/// <param name="min">		The minimum. </param>
/// <param name="max">		The maximum. </param>
///
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
///-------------------------------------------------------------------------------------------------
bool inRange(double value, double min, double max);

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Validate if the vector is not empty and the matrices are validate. </summary>
///
/// <param name="matrices">	Vector of Matrix. </param>
///
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
///-------------------------------------------------------------------------------------------------
bool areNotEmpty(const std::vector<Eigen::MatrixXd>& matrices);

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Validates if matrix is not empty. </summary>
/// 
/// <param name="matrix">	Matrix. </param>
///
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
///-------------------------------------------------------------------------------------------------
bool isNotEmpty(const Eigen::MatrixXd& matrix);

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Validates if two matrix have same size. </summary>
/// 
/// <param name="a">	Matrix A. </param>
/// <param name="b">	Matrix B. </param>
///
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
///-------------------------------------------------------------------------------------------------
bool haveSameSize(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b);

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Validates if matrix is square matrix and not empty. </summary>
/// 
/// <param name="matrix">	Matrix. </param>
///
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
///-------------------------------------------------------------------------------------------------
bool isSquare(const Eigen::MatrixXd& matrix);

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Validate if the vector is not empty and the matrices are square matrix and not empty. </summary>
///
/// <param name="matrices">	Vector of Matrix. </param>
///
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
///-------------------------------------------------------------------------------------------------
bool areSquare(const std::vector<Eigen::MatrixXd>& matrices);

//***************************************************
//***************************************************
//***************************************************


//********************************************************
//******************** CSV MANAGEMENT ********************
//********************************************************
///----------------------------------------------------------------------------------------------------
/// 
/// <summary>Return the string split by the \p sep parameter</summary>
/// 
/// <param name="s">	The string to split.</param>
/// <param name="sep">	the separator string which splits.</param>
/// 
/// <returns>	Vector of string part. </returns>
/// 
///----------------------------------------------------------------------------------------------------
std::vector<std::string> Split(const std::string& s, const std::string& sep);

//********************************************************
//********************************************************
//********************************************************
