///-------------------------------------------------------------------------------------------------
/// 
/// \file Classification.hpp
/// \brief All functions to help Matrix Classifiers.
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 26/10/2018.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// \remarks 
/// - LSQR inspired by <a href="http://scikit-learn.org">sklearn</a> <a href="https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html">LinearDiscriminantAnalysis</a> (<a href="https://github.com/scikit-learn/scikit-learn/blob/master/COPYING">License</a>).
/// - FgDA inspired by the work of Alexandre Barachant : <a href="https://github.com/alexandrebarachant/pyRiemann">pyRiemann</a> (<a href="https://github.com/alexandrebarachant/pyRiemann/blob/master/LICENSE">License</a>).
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include <vector>
#include <Eigen/Dense>

namespace Geometry {

/// <summary>	 Compute the weight of Linear Discriminant Analysis with Least squares (LSQR) Solver. </summary>
/// <param name="datasets">	The datasets one class by row and trials on colums. </param>
/// <param name="weight">	The Weight to apply. </param>
/// <returns>	<c>True</c> if it succeeds, <c>false</c> otherwise. </returns>
/// <remarks>	Inspired by <a href="http://scikit-learn.org">sklearn</a> <a href="https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html">LinearDiscriminantAnalysis</a> (<a href="https://github.com/scikit-learn/scikit-learn/blob/master/COPYING">License</a>).</remarks>
bool LSQR(const std::vector<std::vector<Eigen::RowVectorXd>>& datasets, Eigen::MatrixXd& weight);

/// <summary>	 Compute Least squares (LSQR) Weight and transform to FgDA Weight. \n
///	\f[ W_{\text{FgDA}} = W^{\mathsf{T}} \times (W \times W^{\mathsf{T}})^{-1} \times W \f]
/// </summary>
/// <param name="datasets">	The data set one class by row and trials on colums. </param>
/// <param name="weight">	The Weight to apply. </param>
/// <returns>	<c>True</c> if it succeeds, <c>false</c> otherwise. </returns>
/// <remarks>	Method inspired by the work of Alexandre Barachant : <a href="https://github.com/alexandrebarachant/pyRiemann">pyRiemann</a> (<a href="https://github.com/alexandrebarachant/pyRiemann/blob/master/LICENSE">License</a>).</remarks>
bool FgDACompute(const std::vector<std::vector<Eigen::RowVectorXd>>& datasets, Eigen::MatrixXd& weight);

/// <summary>	 Apply the weight on the vector. (just a matrix product) </summary>
/// <param name="in">		Sample to transform. </param>
/// <param name="out">		Transformed Sample. </param>
/// <param name="weight">	The Weight to apply. </param>
/// <returns>	<c>True</c> if it succeeds, <c>false</c> otherwise. </returns>
/// <remarks>	Method inspired by the work of Alexandre Barachant : <a href="https://github.com/alexandrebarachant/pyRiemann">pyRiemann</a> (<a href="https://github.com/alexandrebarachant/pyRiemann/blob/master/LICENSE">License</a>).</remarks>
bool FgDAApply(const Eigen::RowVectorXd& in, Eigen::RowVectorXd& out, const Eigen::MatrixXd& weight);

}  // namespace Geometry
