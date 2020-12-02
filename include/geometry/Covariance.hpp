///-------------------------------------------------------------------------------------------------
/// 
/// \file Covariance.hpp
/// \brief All functions to estimate the Covariance Matrix.
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 26/10/2018.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// \remarks 
/// - List of Estimator inspired by the work of Alexandre Barachant : <a href="https://github.com/alexandrebarachant/pyRiemann">pyRiemann</a> (<a href="https://github.com/alexandrebarachant/pyRiemann/blob/master/LICENSE">License</a>).
/// - <a href="http://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html">Ledoit and Wolf Estimator</a> inspired by <a href="http://scikit-learn.org">sklearn</a> (<a href="https://github.com/scikit-learn/scikit-learn/blob/master/COPYING">License</a>).
/// - <a href="http://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html">Oracle Approximating Shrinkage (OAS) Estimator</a> Inspired by <a href="http://scikit-learn.org">sklearn</a> (<a href="https://github.com/scikit-learn/scikit-learn/blob/master/COPYING">License</a>).
/// - <b>Minimum Covariance Determinant (MCD) Estimator isn't implemented. </b>
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include "geometry/Basics.hpp"
#include <Eigen/Dense>

namespace Geometry {

//***************************************************
//******************** CONSTANTS ********************
//***************************************************
/// <summary>	Enumeration of the covariance matrix estimator. Inspired by the work of Alexandre Barachant : <a href="https://github.com/alexandrebarachant/pyRiemann">pyRiemann</a>. </summary>
enum class EEstimator
{
	COV,	///< The Simple Covariance Estimator.
	SCM,	///< The Normalized Spatial Covariance Matrix (SCM) Estimator.
	LWF,	///< The Ledoit and Wolf Estimator.
	OAS,	///< The Oracle Approximating Shrinkage (OAS) Estimator.
	MCD,	///< The Minimum Covariance Determinant (MCD) Estimator.
	COR,	///< The Pearson Correlation Estimator.
	IDE		///< The Identity Matrix.
};

/// <summary>	Convert estimators to string. </summary>
/// <param name="estimator">	The estimator. </param>
/// <returns>	<c>std::string</c> </returns>
inline std::string toString(const EEstimator estimator)
{
	switch (estimator)
	{
		case EEstimator::COV: return "Covariance";
		case EEstimator::SCM: return "Normalized Spatial Covariance Matrix (SCM)";
		case EEstimator::LWF: return "Ledoit and Wolf";
		case EEstimator::OAS: return "Oracle Approximating Shrinkage (OAS)";
		case EEstimator::MCD: return "Minimum Covariance Determinant (MCD)";
		case EEstimator::COR: return "Pearson Correlation";
		case EEstimator::IDE: return "Identity";
	}
	return "Invalid";
}

/// <summary>	Convert string to estimators. </summary>
/// <param name="estimator">	The estimator. </param>
/// <returns>	<see cref="EEstimator"/> </returns>
inline EEstimator StringToEstimator(const std::string& estimator)
{
	if (estimator == "Covariance") { return EEstimator::COV; }
	if (estimator == "Normalized Spatial Covariance Matrix (SCM)") { return EEstimator::SCM; }
	if (estimator == "Ledoit and Wolf") { return EEstimator::LWF; }
	if (estimator == "Oracle Approximating Shrinkage (OAS)") { return EEstimator::OAS; }
	if (estimator == "Minimum Covariance Determinant (MCD)") { return EEstimator::MCD; }
	if (estimator == "Pearson Correlation") { return EEstimator::COR; }
	return EEstimator::IDE;
}

//***********************************************************
//******************** COVARIANCES BASES ********************
//***********************************************************
/// <summary>	Calculation of the Variance of a double dataset \f$\vec{X}\f$.\n
/// \f[  V(X) = \left(\frac{1}{n} \sum_{i=1}^{N}x_{i}^{2}\right) - \left(\frac{1}{n} \sum_{i=1}^{N}x_{i}\right)^{2} \f]
/// </summary>
/// <param name="x">	The dataset \f$\vec{X}\f$. With \f$ N \f$ Samples. </param>
/// <returns>	The Variance. </returns>
double Variance(const Eigen::RowVectorXd& x);

/// <summary>	Calculation of the Covariance between two double dataset \f$\vec{X}, \vec{Y}\f$.\n
/// \f[ \operatorname{Cov}\left(x,y\right) = \frac{\sum_{i=1}^{N}{x_{i}y_{i}} - \left(\sum_{i=1}^{N}{x_{i}}\sum_{i=1}^{N}{y_{i}}\right)/N}{N}\f]
/// </summary>
/// <param name="x">	The dataset \f$\vec{X}\f$. With \f$ N \f$ Samples. </param>
/// <param name="y">	The dataset \f$\vec{Y}\f$. With \f$ N \f$ Samples. </param>
/// <returns>	The Covariance. </returns>
double Covariance(const Eigen::RowVectorXd& x, const Eigen::RowVectorXd& y);

/// <summary>	Shrunks the Covariance Matrix \f$ M \f$ (destructive operation).\n
/// \f[ (1 - \text{shrinkage}) \times M_{\operatorname{Cov}} + \frac{\text{shrinkage} \times \operatorname{trace}(M_{Cov})}{N} \times I_N \f]
/// </summary>
/// <param name="cov">			The Covariance Matrix to shrink. </param>
/// <param name="shrinkage">	(Optional) The shrinkage coefficient : \f$ 0\leq \text{shrinkage} \leq 1\f$. </param>
/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
bool ShrunkCovariance(Eigen::MatrixXd& cov, double shrinkage = 0.1);

/// <summary>	Shrunks the Covariance Matrix \f$ M \f$ (non destructive operation).\n
/// \f[ (1 - \text{shrinkage}) \times M_{\operatorname{Cov}} + \frac{\text{shrinkage} \times \operatorname{trace}(M_{Cov})}{N} \times I_N \f]
/// </summary>
/// <param name="in">			The covariance matrix to shrink. </param>
/// <param name="out">			The shrunk covariance matrix. </param>
/// <param name="shrinkage">	(Optional) The shrinkage coefficient : \f$ 0\leq \text{shrinkage} \leq 1\f$. </param>
/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
bool ShrunkCovariance(const Eigen::MatrixXd& in, Eigen::MatrixXd& out, double shrinkage = 0.1);

/// <summary>	Select the function to call for the covariance matrix.\n
/// - centralizing the data is useless for <c><see cref="EEstimator::COV"/></c> and <c><see cref="EEstimator::COR"/></c>.\n
/// - centralizing the data is not usual for <c><see cref="EEstimator::SCM"/></c>.
/// </summary>
/// <param name="in">			The dataset \f$\vec{X}\f$. With \f$ N \f$ Rows (features) and \f$ S \f$ columns (samples). </param>
/// <param name="out">			The Covariance Matrix. </param>
/// <param name="estimator">	(Optional) The selected estimator (see <see cref="EEstimator"/>). </param>
/// <param name="standard">		(Optional) Standardize the data (see <see cref="EStandardization"/>). </param>
/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
bool CovarianceMatrix(const Eigen::MatrixXd& in, Eigen::MatrixXd& out, EEstimator estimator = EEstimator::COV,
					  EStandardization standard                                             = EStandardization::Center);

//***********************************************************
//******************** COVARIANCES TYPES ********************
//***********************************************************
/// <summary>	Calculation of the covariance matrix.\n
/// \f[ M_{\operatorname{Cov}} =
///		\begin{pmatrix}
///			V\left(x_1\right) & \operatorname{Cov}\left(x_1,x_2\right) &\cdots & \operatorname{Cov}\left(x_1,x_N\right)\\
///			\operatorname{Cov}\left(x_2,x_1\right) &\ddots & \ddots & \vdots \\
///			\vdots & \ddots & \ddots & \vdots \\
///			\operatorname{Cov}\left(x_N,x_1\right) &\cdots & \cdots & V\left(x_N\right)
///		\end{pmatrix}
///		\quad\quad \text{with } x_i \text{ the feature } i
///	\f]\n
///	With the <see cref="Variance"/> and <see cref="Covariance"/> function.
/// </summary>
/// <param name="samples">	The dataset \f$\vec{X}\f$. With \f$ N \f$ Rows (features) and \f$ S \f$ columns (samples). </param>
/// <param name="cov">	  	The Covariance Matrix. </param>
/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
bool CovarianceMatrixCOV(const Eigen::MatrixXd& samples, Eigen::MatrixXd& cov);

/// <summary>	Calculation of the covariance matrix by the method : Normalized Spatial Covariance Matrix (SCM).\n
///	\f[ M_{\operatorname{Cov_{SCM}}} = \frac{XX^{\mathsf{T}}}{\operatorname{trace}{\left(XX^{\mathsf{T}}\right)}} \f]
/// </summary>
/// <param name="samples">	The dataset \f$\vec{X}\f$. With \f$ N \f$ Rows (features) and \f$ S \f$ columns (samples). </param>
/// <param name="cov">	  	The Covariance Matrix. </param>
/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
bool CovarianceMatrixSCM(const Eigen::MatrixXd& samples, Eigen::MatrixXd& cov);

/// <summary>	Calculation of the covariance matrix and shrinkage by the method : Ledoit and Wolf.\n
/// -# Compute the Covariance Matrix (see <see cref="CovarianceMatrixCOV"/>) \f$ M_{\operatorname{Cov}} \f$
/// -# Compute the Ledoit and Wolf Shrinkage
/// -# Shrunk the Matrix (see <see cref="ShrunkCovariance"/>)
/// 
/// Ledoit and Wolf Shrinkage (from <a href="http://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html">Sklearn LedoitWolf Estimator</a>) 
/// described in "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices", Ledoit and Wolf, Journal of Multivariate Analysis, Volume 88, Issue 2, February 2004, pages 365-411. : \n
/// \f[ 
///		\begin{aligned}
///			\vec{X}^2 &= \begin{pmatrix}x_{0,0}^2 & \cdots & x_{0,S}^2 \\ \vdots & \ddots &\vdots \\ x_{N,0}^2 & \cdots & x_{N,S}^2\end{pmatrix} \quad \text{with } x_{i,j} \in \vec{X}\\
///			M_{\mu} &= \mu\times I_N =  \begin{pmatrix} \mu & 0 & \cdots & 0 \\ 0 & \ddots &\ddots & \vdots \\ \vdots & \ddots &\ddots & 0 \\ 0 & \cdots & 0 & \mu\end{pmatrix} 
/// 													\quad \text{with } \mu = \frac{\operatorname{trace}(M_{\operatorname{Cov}})}{N}\\
/// 		 M_{\delta} &= M_{\operatorname{Cov}}-M_{\mu}\\
/// 		 M_{\delta}^2 &= M_{\delta} * M_{\delta}\\
/// 		 M_{\beta} &= \frac{1}{S} \times \left(\vec{X}^2 * \vec{X}^{2\mathsf{T}}\right) - M_{Cov} * M_{Cov}\\
/// 		 \Sigma\left( M \right) &=\text{ the sum of the elements of the matrix } M\\
/// 	\end{aligned}
/// \f]
/// \f[ \text{Shrinkage}_\text{LWF} = \frac{\beta}{\delta} \quad \text{with } \delta = \frac{\Sigma\left( M_{\delta}^2 \right)}{N} \quad\text{and}\quad 
/// 																				  \beta = \operatorname{min}\left(\frac{\Sigma\left( M_{\beta}^2 \right)}{N \times S},~ \delta\right)\f]
/// </summary>
/// <param name="samples">	The dataset \f$\vec{X}\f$. With \f$ N \f$ Rows (features) and \f$ S \f$ columns (samples). </param>
/// <param name="cov">	  	The Covariance Matrix. </param>
/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
bool CovarianceMatrixLWF(const Eigen::MatrixXd& samples, Eigen::MatrixXd& cov);

/// <summary> Calculation of the covariance matrix and shrinkage by the method : Oracle Approximating Shrinkage (OAS).\n
/// -# Compute the Covariance Matrix (see <see cref="CovarianceMatrixCOV"/>) \f$ M_{\operatorname{Cov}} \f$
/// -# Compute the Oracle Approximating Shrinkage
/// -# Shrunk the Matrix (see <see cref="ShrunkCovariance"/>)
/// 
/// Oracle Approximating Shrinkage (from <a href="http://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html">Sklearn Oracle Approximating Shrinkage Estimator</a>) 
/// describe in "Shrinkage Algorithms for MMSE Covariance Estimation" Chen et al., IEEE Trans. on Sign. Proc., Volume 58, Issue 10, October 2010. : \n
/// \f[ 
///		\begin{aligned}
///			\mu &= \frac{\operatorname{trace}(M_{\operatorname{Cov}})}{N}\\
///			\mu \left( M \right) &=\text{ the mean of the elements of the matrix } M\\
///			\alpha &= \mu \left( M_{\operatorname{Cov}} * M_{\operatorname{Cov}} \right)\\
///			\text{num} &= \alpha + \mu^2\\
///			\text{den} &= (S + 1) \times \frac{\alpha - \mu^2}{N}\\
///		\end{aligned}
/// \f]
/// \f[
///		\text{Shrinkage}_\text{OAS} =	\begin{cases}
///											1, & \text{if}\ \text{den} = 0 \text{ or num} > \text{den} \\
///											\frac{\text{num}}{\text{den}}, & \text{otherwise}
///										\end{cases}
///	\f]
/// </summary>
/// <param name="samples">	The dataset \f$\vec{X}\f$. With \f$ N \f$ Rows (features) and \f$ S \f$ columns (samples). </param>
/// <param name="cov">	  	The Covariance Matrix. </param>
/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
bool CovarianceMatrixOAS(const Eigen::MatrixXd& samples, Eigen::MatrixXd& cov);

/// <summary>Calculation of the covariance matrix and shrinkage by the method : Minimum Covariance Determinant (MCD). </summary>
/// <param name="samples">	The dataset \f$\vec{X}\f$. With \f$ N \f$ Rows (features) and \f$ S \f$ columns (samples). </param>
/// <param name="cov">	  	The Covariance Matrix. </param>
/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
/// \todo Not implemented.
bool CovarianceMatrixMCD(const Eigen::MatrixXd& samples, Eigen::MatrixXd& cov);

/// <summary>	Calculation of the covariance matrix by the method : Pearson Correlation.\n
/// \f[
///		M_{\operatorname{Cov_{COR}}}\left(i,j\right) 
///		= \frac{  M_{\operatorname{Cov}}\left(i,j\right) } { \sqrt{  M_{\operatorname{Cov}}\left(i,i\right) *  M_{\operatorname{Cov}}\left(j,j\right) } }
///	\f]
/// </summary>
/// <param name="samples">	The dataset \f$\vec{X}\f$. With \f$ N \f$ Rows (features) and \f$ S \f$ columns (samples). </param>
/// <param name="cov">	  	The Covariance Matrix. </param>
/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
bool CovarianceMatrixCOR(const Eigen::MatrixXd& samples, Eigen::MatrixXd& cov);

/// <summary> Return the Identity matrix \f$ I_N \f$. </summary>
/// <param name="samples">	The dataset \f$\vec{X}\f$. With \f$ N \f$ Rows (features) and \f$ S \f$ columns (samples). </param>
/// <param name="cov">	  	The Covariance Matrix. </param>
/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
bool CovarianceMatrixIDE(const Eigen::MatrixXd& samples, Eigen::MatrixXd& cov);

}  // namespace Geometry
