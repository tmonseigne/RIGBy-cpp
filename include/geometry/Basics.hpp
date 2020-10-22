///-------------------------------------------------------------------------------------------------
/// 
/// \file Basics.hpp
/// \brief Basic functions of Eigen matrix manipulation and verification.
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 26/10/2018.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>		// Ceil
#include <type_traits>	// Template type

/// <summary> Enumeration of Standardization method for features matrix data.</summary>
enum class EStandardization
{
	None,			///< No change.
	Center,			///< Standardize data by removing the mean (on each feature separately).
	StandardScale	///< Standardize data by removing the mean and scaling to unit variance (on each feature separately).
};

//************************************************
//******************** Matrix ********************
//************************************************

/// <summary>	Apply an affine transformation and return the result (The last transpose is useless if matrix is SPD).
/// \f[
/// B = R^{-1/2} * A * {R^{-1/2}}^{\mathsf{T}}
/// \f]\n
/// </summary>
/// <param name="ref"> 	The reference matrix which transforms. </param>
/// <param name="matrix">	the matrix to transform. </param>
/// <returns>	The transformed matrix </returns>
Eigen::MatrixXd AffineTransformation(const Eigen::MatrixXd& ref, const Eigen::MatrixXd& matrix);


/// <summary>	Standardize data row by row with selected method (destructive operation). </summary>
/// <param name="matrix"> 	The matrix to standardize. </param>
/// <param name="standard">	Standard method. </param>
/// <returns>	True if it succeeds, false if it fails. </returns>
bool MatrixStandardization(Eigen::MatrixXd& matrix, EStandardization standard = EStandardization::None);

/// <summary>	Standardize data row by row with selected method (non destructive operation). </summary>
/// <param name="in"> 		The matrix to standardize. </param>
/// <param name="out">		The matrix standardized. </param>
/// <param name="standard">	Standard method. </param>
/// <returns>	True if it succeeds, false if it fails. </returns>
bool MatrixStandardization(const Eigen::MatrixXd& in, Eigen::MatrixXd& out, EStandardization standard = EStandardization::None);

/// <summary>	Removes the mean of each row at the matrix (destructive operation). </summary>
/// <param name="matrix">	The Matrix to center. </param>
/// <returns>	True if it succeeds, false if it fails. </returns>
bool MatrixCenter(Eigen::MatrixXd& matrix);

/// <summary>	Removes the mean of each row at the matrix (non destructive operation). </summary>
/// <param name="in">	The Matrix to center. </param>
/// <param name="out">	The Matrix centered. </param>
/// <returns>	True if it succeeds, false if it fails. </returns>
bool MatrixCenter(const Eigen::MatrixXd& in, Eigen::MatrixXd& out);

/// <summary>	Removes the mean of each row at the matrix and divide by the variance (destructive operation with scale return). </summary>
/// <param name="matrix">	The Matrix to standardize. </param>
/// <param name="scale">	The scale vector. </param>
/// <returns>	True if it succeeds, false if it fails. </returns>
/// <remarks> Adaptation of <a href="http://scikit-learn.org">sklearn</a> <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">StandardScaler</a> (<a href="https://github.com/scikit-learn/scikit-learn/blob/master/COPYING">License</a>).</remarks>
bool MatrixStandardScaler(Eigen::MatrixXd& matrix, Eigen::RowVectorXd& scale);

/// <summary>	Removes the mean of each row at the matrix and divide by the variance (destructive operation). </summary>
/// <param name="matrix">	The Matrix to standardize. </param>
/// <returns>	True if it succeeds, false if it fails. </returns>
/// <remarks> Adaptation of <a href="http://scikit-learn.org">sklearn</a> <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">StandardScaler</a> (<a href="https://github.com/scikit-learn/scikit-learn/blob/master/COPYING">License</a>).</remarks>
bool MatrixStandardScaler(Eigen::MatrixXd& matrix);

/// <summary>	Removes the mean of each row at the matrix and divide by the variance (non destructive operation). </summary>
/// <param name="in">		The Matrix to standardize. </param>
/// <param name="out">		The Matrix standardized. </param>
/// <param name="scale">	The scale vector. </param>
/// <returns>	True if it succeeds, false if it fails. </returns>
/// <remarks> Adaptation of <a href="http://scikit-learn.org">sklearn</a> <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">StandardScaler</a> (<a href="https://github.com/scikit-learn/scikit-learn/blob/master/COPYING">License</a>).</remarks>
bool MatrixStandardScaler(const Eigen::MatrixXd& in, Eigen::MatrixXd& out, Eigen::RowVectorXd& scale);

/// <summary>	Removes the mean of each row at the matrix and divide by the variance (non destructive operation). </summary>
/// <param name="in">	The Matrix to standardize. </param>
/// <param name="out">	The Matrix standardized. </param>
/// <returns>	True if it succeeds, false if it fails. </returns>
/// <remarks> Adaptation of <a href="http://scikit-learn.org">sklearn</a> <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">StandardScaler</a> (<a href="https://github.com/scikit-learn/scikit-learn/blob/master/COPYING">License</a>).</remarks>
bool MatrixStandardScaler(const Eigen::MatrixXd& in, Eigen::MatrixXd& out);

/// <summary>	Give the string format of Matrix for OpenViBE Log. </summary>
/// <param name="matrix">	The Matrix to display. </param>
/// <returns>		The string format. </returns>
std::string MatrixPrint(const Eigen::MatrixXd& matrix);

/// <summary>	Check first the size, then if not empty matrix and then if they are almost equal. </summary>
/// <param name="matrix1">		First Matrix. </param>
/// <param name="matrix2">		Second Matrix. </param>
/// <param name="precision">	Precision for matrix comparison. </param>
/// <returns>		True if Equals, false if not. </returns>
bool AreEquals(const Eigen::MatrixXd& matrix1, const Eigen::MatrixXd& matrix2, double precision = 1e-6);

//*************************************************************
//******************** Index Manipulations ********************
//*************************************************************
/// <summary>	Gets the items selected by the index. </summary>
/// <param name="row">		the original row. </param>
/// <param name="index">	Elements to select. </param>
/// <returns>	Row with selected elements. </returns>
Eigen::RowVectorXd GetElements(const Eigen::RowVectorXd& row, const std::vector<size_t>& index);

/// <summary>	Numpy arange in C++. </summary>
/// <typeparam name="T">	Generic numeric type parameter. </typeparam>
/// <param name="start">	The start. </param>
/// <param name="stop"> 	The stop. </param>
/// <param name="step"> 	(Optional) Amount to increment by. </param>
/// <returns>	vector&lt;T&gt; </returns>
template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
std::vector<T> ARange(const T start, const T stop, const T step = 1)
{
	std::vector<T> result;
	result.reserve(size_t(ceil(1.0 * (stop - start) / step)));
	for (T i = start; i < stop; i += step) { result.push_back(i); }
	return result;
}

/// <summary>	Turn vector of vector into vector. </summary>
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="in">	vector of vector. </param>
/// <returns>	vector&lt;T&gt; </returns>
template <typename T>
std::vector<T> Vector2DTo1D(const std::vector<std::vector<T>>& in)
{
	std::vector<T> result;
	size_t sum = 0;
	for (const auto& v : in) { sum += v.size(); }
	result.reserve(sum);
	for (const auto& v : in) { for (const auto& e : v) { result.push_back(e); } }
	return result;
}

/// <summary>	Turn vector into vector of vector with postiion repartition. </summary>
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="in">		vector of vector. </param>
/// <param name="position">	position of element (size of position is the number of row the values are the number of element on each row). </param>
/// <returns>	vector&lt;T&gt; </returns>
template <typename T>
std::vector<std::vector<T>> Vector1DTo2D(const std::vector<T>& in, const std::vector<size_t>& position)
{
	const size_t n = position.size();
	std::vector<std::vector<T>> result(n);
	size_t idx = 0;
	for (size_t i = 0; i < n; ++i)
	{
		const size_t nbSample = position[i];
		result[i].resize(nbSample);
		for (size_t j = 0; j < nbSample; ++j) { result[i][j] = in[idx++]; }
	}
	return result;
}

//***************************************************
//******************** Validates ********************
//***************************************************
/// <summary>	Validate if value is in [min;max]. </summary>
/// <param name="value">	The value. </param>
/// <param name="min">		The minimum. </param>
/// <param name="max">		The maximum. </param>
/// <returns>	True if it succeeds, false if it fails. </returns>
bool InRange(double value, double min, double max);

/// <summary>	Validate if the vector is not empty and the matrices are validate. </summary>
/// <param name="matrices">	Vector of Matrix. </param>
/// <returns>	True if it succeeds, false if it fails. </returns>
bool AreNotEmpty(const std::vector<Eigen::MatrixXd>& matrices);

/// <summary>	Validates if matrix is not empty. </summary>
/// <param name="matrix">	Matrix. </param>
/// <returns>	True if it succeeds, false if it fails. </returns>
bool IsNotEmpty(const Eigen::MatrixXd& matrix);

/// <summary>	Validates if two matrix have same size. </summary>
/// <param name="a">	Matrix A. </param>
/// <param name="b">	Matrix B. </param>
/// <returns>	True if it succeeds, false if it fails. </returns>
bool HaveSameSize(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b);

/// <summary>	Validate if the vector is not empty and the matrices have same size. </summary>
/// <param name="matrices">	Vector of Matrix. </param>
/// <returns>	True if it succeeds, false if it fails. </returns>
bool HaveSameSize(const std::vector<Eigen::MatrixXd>& matrices);

/// <summary>	Validates if matrix is square matrix and not empty. </summary>
/// <param name="matrix">	Matrix. </param>
/// <returns>	True if it succeeds, false if it fails. </returns>
bool IsSquare(const Eigen::MatrixXd& matrix);

/// <summary>	Validate if the vector is not empty and the matrices are square matrix and not empty. </summary>
/// <param name="matrices">	Vector of Matrix. </param>
/// <returns>	True if it succeeds, false if it fails. </returns>
bool AreSquare(const std::vector<Eigen::MatrixXd>& matrices);

//********************************************************
//******************** CSV MANAGEMENT ********************
//********************************************************
/// <summary>Return the string split by the \p sep parameter</summary>
/// <param name="s">	The string to split.</param>
/// <param name="sep">	the separator string which splits.</param>
/// <returns>	Vector of string part. </returns>
std::vector<std::string> Split(const std::string& s, const std::string& sep);
