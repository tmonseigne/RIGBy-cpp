///-------------------------------------------------------------------------------------------------
/// 
/// \file Covariance.hpp
/// 
/// \brief All functions to estimate the Covariance Matrix.
/// 
/// \author Thibaut Monseigne (Inria).
/// 
/// \version 1.0.
/// 
/// \date 26/10/2018.
/// 
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// 
/// \remarks 
/// - List of Estimator inspired by the work of Alexandre Barachant : <a href="https://github.com/alexandrebarachant/pyRiemann">pyRiemann</a> (<a href="https://github.com/alexandrebarachant/pyRiemann/blob/master/LICENSE">License</a>).
/// - <a href="http://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html">Ledoit and Wolf Estimator</a> inspired by <a href="http://scikit-learn.org">sklearn</a> (<a href="https://github.com/scikit-learn/scikit-learn/blob/master/COPYING">License</a>).
/// - <a href="http://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html">Oracle Approximating Shrinkage (OAS) Estimator</a> Inspired by <a href="http://scikit-learn.org">sklearn</a> (<a href="https://github.com/scikit-learn/scikit-learn/blob/master/COPYING">License</a>) doesn't Work.
/// - Normalized Spatial Covariance Matrix (SCM) Estimator must be validated by another library.
/// - Minimum Covariance Determinant (MCD) Estimator isn't implemented.
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>

//***************************************************
//******************** CONSTANTS ********************
//***************************************************
/// <summary> Enumeration of the covariance matrix estimator. Inspired by the work of Alexandre Barachant : <a href="https://github.com/alexandrebarachant/pyRiemann">pyRiemann</a>. </summary>
enum EEstimator
{
	/// <summary>The Simple Covariance Estimator.</summary>
	Estimator_COV,
	/// <summary>The Sample Covariance Matrix (SCM) Estimator.</summary>
	Estimator_SCM,
	/// <summary>The Ledoit and Wolf Estimator.</summary>
	Estimator_LWF,
	/// <summary>The Oracle Approximating Shrinkage (OAS) Estimator.</summary>
	Estimator_OAS,
	/// <summary>The Minimum Covariance Determinant (MCD) Estimator.</summary>
	Estimator_MCD,
	/// <summary>The Pearson Correlation Estimator.</summary>
	Estimator_COR,
	/// <summary>The Identity Matrix.</summary>
	Estimator_IDE
};

inline std::string EstimatorToString(const EEstimator estimator)
{
	switch (estimator)
	{
		case Estimator_COV: return "Covariance";
		case Estimator_SCM: return "Sample Covariance Matrix (SCM)";
		case Estimator_LWF: return "Ledoit and Wolf";
		case Estimator_OAS: return "Oracle Approximating Shrinkage (OAS)";
		case Estimator_MCD: return "Minimum Covariance Determinant (MCD)";
		case Estimator_COR: return "Pearson Correlation";
		case Estimator_IDE: return "Identity";
		default: return "Invalid";
	}
}
//***************************************************
//***************************************************
//***************************************************

//***********************************************************
//******************** COVARIANCES BASES ********************
//***********************************************************
///----------------------------------------------------------------------------------------------------
/// 
/// <summary>	Calculation of the Variance of a double data set \f$\vec{X}\f$.
///
/// \f[  V(X) = \left(\frac{1}{n} \sum_{i=1}^{N}x_{i}^{2}\right) - \mu^{2} \quad \text{with}~ \mu = \frac{1}{n} \sum_{i=1}^{N}x_{i} \f]
/// </summary>
/// 
/// <param name="x">The data set \f$\vec{X}\f$. With \f$ N \f$ Samples.</param>
/// 
/// <returns> The Variance. </returns>
/// 
///-------------------------------------------------------------------------------------------------
double Variance(const Eigen::RowVectorXd& x);

///----------------------------------------------------------------------------------------------------
/// 
/// <summary>	Calculation of the Covariance between two double data set \f$\vec{X}\f$, \f$\vec{Y}\f$.
/// 			
/// \f[ \operatorname{Cov}\left(x,y\right) = \frac{\sum_{i=1}^{N}{x_{i}y_{i}} - \left(\sum_{i=1}^{N}{x_{i}}\sum_{i=1}^{N}{y_{i}}\right)/N}{N}\f]
/// </summary>
/// 
/// <param name="x">	The data set \f$\vec{X}\f$. With \f$ N \f$ Samples.</param>
/// <param name="y">	The data set \f$\vec{Y}\f$. With \f$ N \f$ Samples.</param>
/// 
/// <returns>	The Covariance. </returns>
/// 
///-------------------------------------------------------------------------------------------------
double Covariance(const Eigen::RowVectorXd& x, const Eigen::RowVectorXd& y);

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Shrunks the Covariance Matrix \f$ M \f$ (destructive operation).
/// 			
/// \f[ (1 - \text{shrinkage}) \times M_{\operatorname{Cov}} + \frac{\text{shrinkage} \times \operatorname{trace}(M_{Cov})}{N} \times I_N \f]
/// </summary>
/// 
/// <param name="cov">			The Covariance Matrix to shrink. </param>
/// <param name="shrinkage">	(Optional) The shrinkage coefficient : \f$ 0\leq \text{shrinkage} \leq 1\f$. </param>
/// 
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
///----------------------------------------------------------------------------------------------------
bool ShrunkCovariance(Eigen::MatrixXd& cov, double shrinkage = 0.1);

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Shrunks the Covariance Matrix \f$ M \f$ (non destructive operation).
/// 	
/// \f[ (1 - \text{shrinkage}) \times M_{\operatorname{Cov}} + \frac{\text{shrinkage} \times \operatorname{trace}(M_{Cov})}{N} \times I_N \f]
/// </summary>
/// 
/// <param name="in">			The covariance matrix to shrink. </param>
/// <param name="out">			The shrunk covariance matrix. </param>
/// <param name="shrinkage">	(Optional) The shrinkage coefficient : \f$ 0\leq \text{shrinkage} \leq 1\f$. </param>
/// 
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
///----------------------------------------------------------------------------------------------------
bool ShrunkCovariance(const Eigen::MatrixXd& in, Eigen::MatrixXd& out, double shrinkage = 0.1);

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Select the function to call for the covariance matrix.
/// 			
/// - centralizing the data is useless for <see cref="Estimator_COV"/> and <see cref="Estimator_COR"/>.\n
/// - centralizing the data is not usual for <see cref="Estimator_SCM"/>.
/// </summary>
/// 
/// <param name="in">			The data set \f$\vec{X}\f$. With \f$ N \f$ Rows (features) and \f$ S \f$ columns (samples). </param>
/// <param name="out">			The Covariance Matrix. </param>
/// <param name="estimator">	(Optional) The selected estimator (see <see cref="EEstimator"/>). </param>
/// <param name="center">   	(Optional) True to center the datas (each row is centered separately). </param>
/// 
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
///----------------------------------------------------------------------------------------------------
bool CovarianceMatrix(const Eigen::MatrixXd& in, Eigen::MatrixXd& out, EEstimator estimator = Estimator_COV, bool center = true);
//***********************************************************
//***********************************************************
//***********************************************************

//***********************************************************
//******************** COVARIANCES TYPES ********************
//***********************************************************
///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Calculation of the covariance matrix.
/// 			
/// \f[ M_{\operatorname{Cov}} =
///		\begin{pmatrix}
///			V\left(x_1\right) & \operatorname{Cov}\left(x_1,x_2\right) &\cdots & \operatorname{Cov}\left(x_1,x_N\right)\\
///			\operatorname{Cov}\left(x_2,x_1\right) &\ddots & \cdots & \vdots \\
///			\vdots & \vdots & \ddots & \vdots \\
///			\operatorname{Cov}\left(x_N,x_1\right) &\cdots & \cdots & V\left(x_N\right)
///		\end{pmatrix}
///		\quad \text{with } x_i \text{ the row } i
///	\f]\n
///	With the <see cref="Variance"/> and <see cref="Covariance"/> function.
/// </summary>
/// 
/// <param name="samples">	The data set \f$\vec{X}\f$. With \f$ N \f$ Rows (features) and \f$ S \f$ columns (samples). </param>
/// <param name="cov">	  	The Covariance Matrix. </param>
/// 
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
///----------------------------------------------------------------------------------------------------
bool CovarianceMatrixCOV(const Eigen::MatrixXd& samples, Eigen::MatrixXd& cov);

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Calculation of the covariance matrix by the method : Normalized Spatial Covariance Matrix (SCM).
/// 			
///	\f[ M_{\operatorname{Cov_{SCM}}} = \frac{XX^{\mathsf{T}}}{\operatorname{trace}{\left(XX^{\mathsf{T}}\right)}} \f]
/// </summary>
/// 
/// <param name="samples">	The data set \f$\vec{X}\f$. With \f$ N \f$ Rows (features) and \f$ S \f$ columns (samples). </param>
/// <param name="cov">	  	The Covariance Matrix. </param>
/// 
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
/// \todo Must be validated by another library.
///
///----------------------------------------------------------------------------------------------------
bool CovarianceMatrixSCM(const Eigen::MatrixXd& samples, Eigen::MatrixXd& cov);

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Calculation of the covariance matrix and shrinkage by the method : Ledoit and Wolf.
/// 			
/// -# Compute the Covariance Matrix (see <see cref="CovarianceMatrixCOV"/>) \f$ M_{\operatorname{Cov}} \f$
/// -# Compute the Ledoit and Wolf Shrinkage
/// -# Shrunk the Matrix (see <see cref="ShrunkCovariance"/>)
/// 
/// Ledoit and Wolf Shrinkage (from <a href="http://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html">Sklearn LedoitWolf Estimator</a>) 
/// 		 described in "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices", Ledoit and Wolf, Journal of Multivariate Analysis, Volume 88, Issue 2, February 2004, pages 365-411. : \n
/// 		 \f[ \begin{align}
/// 		 \vec{X}^2 &= \begin{pmatrix}x_{0,0}^2 & \cdots & x_{0,S}^2 \\ \vdots & \ddots &\vdots \\ x_{N,0}^2 & \cdots & x_{N,S}^2\end{pmatrix} \quad \text{with } x_{i,j} \in \vec{X}\\
/// 		 M_{\mu} &= \mu\times I_N =  \begin{pmatrix} \mu & 0 & \cdots & 0 \\ 0 & \ddots &\ddots & \vdots \\ \vdots & \ddots &\ddots & 0 \\ 0 & \cdots & 0 & \mu\end{pmatrix} 
/// 										\quad \text{with } \mu = \frac{\operatorname{trace}(M_{\operatorname{Cov}})}{N}\\
/// 		 M_{\delta} &= M_{\operatorname{Cov}}-M_{\mu}\\
/// 		 M_{\delta}^2 &= M_{\delta} * M_{\delta}\\
/// 		 M_{\beta} &= \frac{1}{S} \times \left(\vec{X}^2 * \vec{X}^{2\mathsf{T}}\right) - M_{Cov} * M_{Cov}\\
/// 		 \Sigma\left( M \right) &=\text{ the sum of the elements of the matrix } M\\
/// 		 \end{align}\f]
/// 		 \n
/// 		 \f[\text{Shrinkage}_\text{LWF} = \frac{\beta}{\delta} \quad \text{with } \delta = \frac{\Sigma\left( M_{\delta}^2 \right)}{N} \quad\text{and}\quad 
/// 																				  \beta = \operatorname{min}\left(\frac{\Sigma\left( M_{\beta}^2 \right)}{N \times S},~ \delta\right)\f]
/// </summary>
/// 
/// <param name="samples">	The data set \f$\vec{X}\f$. With \f$ N \f$ Rows (features) and \f$ S \f$ columns (samples). </param>
/// <param name="cov">	  	The Covariance Matrix. </param>
/// 
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
///----------------------------------------------------------------------------------------------------
bool CovarianceMatrixLWF(const Eigen::MatrixXd& samples, Eigen::MatrixXd& cov);

///----------------------------------------------------------------------------------------------------
/// 
/// <summary> Calculation of the covariance matrix and shrinkage by the method : Oracle Approximating Shrinkage (OAS).
/// 		 
/// -# Compute the Covariance Matrix (see <see cref="CovarianceMatrixCOV"/>) \f$ M_{\operatorname{Cov}} \f$
/// -# Compute the Oracle Approximating Shrinkage
/// -# Shrunk the Matrix (see <see cref="ShrunkCovariance"/>)
/// 			
/// Oracle Approximating Shrinkage (from <a href="http://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html">Sklearn Oracle Approximating Shrinkage Estimator</a>) 
/// describe in "Shrinkage Algorithms for MMSE Covariance Estimation" Chen et al., IEEE Trans. on Sign. Proc., Volume 58, Issue 10, October 2010. : \n
/// \f[ 
///		\begin{align}
///			\mu &= \frac{\operatorname{trace}(M_{\operatorname{Cov}})}{N}\\
///			\mu \left( M \right) &=\text{ the mean of the elements of the matrix } M\\
///			\alpha &= \mu \left( M_{\operatorname{Cov}} * M_{\operatorname{Cov}} \right)\\
///			\text{num} &= \alpha + \mu^2\\
///			\text{den} &= (S + 1) \times \frac{\alpha - \mu^2}{N}\\
///		\end{align}
/// \f]
/// \f[
///		\text{Shrinkage}_\text{OAS} =	\begin{cases}
///											1, & \text{if}\ \text{den} = 0 \text{ or num} > \text{den} \\
///											\frac{\text{num}}{\text{den}}, & \text{otherwise}
///										\end{cases}
///	\f]
/// </summary>
/// 
/// <param name="samples">	The data set \f$\vec{X}\f$. With \f$ N \f$ Rows (features) and \f$ S \f$ columns (samples). </param>
/// <param name="cov">	  	The Covariance Matrix. </param>
/// 
/// <returns>	True if it succeeds, false if it fails. </returns>

/// \todo Doesn't work
/// 
///----------------------------------------------------------------------------------------------------
bool CovarianceMatrixOAS(const Eigen::MatrixXd& samples, Eigen::MatrixXd& cov);

///----------------------------------------------------------------------------------------------------
/// 
/// <summary>Calculation of the covariance matrix and shrinkage by the method : Minimum Covariance Determinant (MCD).</summary>
/// 
/// <param name="samples">	The data set \f$\vec{X}\f$. With \f$ N \f$ Rows (features) and \f$ S \f$ columns (samples). </param>
/// <param name="cov">	  	The Covariance Matrix. </param>
/// 
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
/// \todo Not implemented.
/// 
///----------------------------------------------------------------------------------------------------
bool CovarianceMatrixMCD(const Eigen::MatrixXd& samples, Eigen::MatrixXd& cov);

///----------------------------------------------------------------------------------------------------
/// 
/// <summary>	Calculation of the covariance matrix by the method : Pearson Correlation.
/// 			
/// \f[
///		M_{\operatorname{Cov_{COR}}}\left(i,j\right) 
///		= \frac{  M_{\operatorname{Cov}}\left(i,j\right) } { \sqrt{  M_{\operatorname{Cov}}\left(i,i\right) *  M_{\operatorname{Cov}}\left(j,j\right) } }
///	\f]
/// </summary>
/// 
/// <param name="samples">	The data set \f$\vec{X}\f$. With \f$ N \f$ Rows (features) and \f$ S \f$ columns (samples). </param>
/// <param name="cov">	  	The Covariance Matrix. </param>
/// 
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
///----------------------------------------------------------------------------------------------------
bool CovarianceMatrixCOR(const Eigen::MatrixXd& samples, Eigen::MatrixXd& cov);

///----------------------------------------------------------------------------------------------------
/// 
/// <summary> Return the Identity matrix \f$ I_N \f$.</summary>
/// 
/// <param name="samples">	The data set \f$\vec{X}\f$. With \f$ N \f$ Rows (features) and \f$ S \f$ columns (samples). </param>
/// <param name="cov">	  	The Covariance Matrix. </param>
/// 
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
///----------------------------------------------------------------------------------------------------
bool CovarianceMatrixIDE(const Eigen::MatrixXd& samples, Eigen::MatrixXd& cov);
//***********************************************************
//***********************************************************
//***********************************************************
