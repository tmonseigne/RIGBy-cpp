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

#include "Metrics.hpp"
#include <Eigen/Dense>
#include <vector>

/// <summary>	Compute the mean of vector of covariance matrix with the selected \p metric. </summary>
/// <param name="covs">  	Vector of Covariance Matrix. </param>
/// <param name="mean">  	The computed mean. </param>
/// <param name="metric">	(Optional) The metric (see <see cref="EMetrics"/>). </param>
/// <returns>	True if it succeeds, false if it fails. </returns>
bool Mean(const std::vector<Eigen::MatrixXd>& covs, Eigen::MatrixXd& mean, EMetrics metric = Metric_Riemann);

/// <summary>	Approximate Joint Diagonalization based on pham's algorithm.\n 
/// \f[ C_\text{AJD} = \cdots \f]
/// </summary>
/// <param name="covs">		Vector of Covariance Matrix. </param>
/// <param name="ajd">	   	The computed Approximate Joint Diagonalization. </param>
/// <param name="epsilon"> 	(Optional) The epsilon. </param>
/// <param name="maxIter">	(Optional) The maximum iterator. </param>
/// <returns>	True if it succeeds, false if it fails. </returns>
/// \todo Not implemented.
bool AJDPham(const std::vector<Eigen::MatrixXd>& covs, Eigen::MatrixXd& ajd, double epsilon = 0.0001, int maxIter = 15);

/// <summary>	Compute the Mean with the Riemannian Mean.\n
/// -# Compute the Classical Mean \f$ C_{\mu_\text{E}} \f$ (see <see cref="MeanEuclidian"/>)
/// -# Update with an iterative procedure that stops after 50 iterations or when one of two criterions is under \f$ 10^{-4}\f$
///
/// \f[ C_{\mu_\text{R}} = C_{\mu_\text{E}} \\ \nu=1.0 \\ \tau=+\infty \f]
/// Iterative process with \f$J\f$ while \f$ \text{iteration} < 50 \f$ and \f$ 10^{-4} < \left\lVert J \right\rVert \f$ and \f$ 10^{-4} < \nu \f$
/// \f[ \begin{aligned}
///		J &= \frac{1}{N} \sum_i \log\left(C_{\mu_\text{R}}^{-1/2} + C_i ~ C_{\mu_\text{R}}^{-1/2}\right)\\
///		C_{\mu_\text{R}} &= C_{\mu_\text{R}}^{1/2} ~ \exp(\nu \times J) ~ C_{\mu_\text{R}}^{1/2}\\
///	\end{aligned}
/// \f]
///	\f[ \begin{cases}
///		\text{if } \nu \times \left\lVert J \right\rVert < \tau & \nu = 0.95 \times \nu,~\tau = \nu \times \left\lVert J \right\rVert\\
///		\text{otherwise } & \nu = 0.5 \times \nu
///	\end{cases}
/// \f]
/// </summary>
/// <param name="covs">	Vector of Covariance Matrix. </param>
/// <param name="mean">	The mean. </param>
/// <returns>	True if it succeeds, false if it fails. </returns>
bool MeanRiemann(const std::vector<Eigen::MatrixXd>& covs, Eigen::MatrixXd& mean);

/// <summary>	Compute the Euclidian Mean.\n
/// \f[ C_{\mu_\text{E}} =\frac{1}{N} \sum_i{C_i}\f]
/// </summary>
/// \copydetails MeanRiemann(const std::vector<Eigen::MatrixXd>&, Eigen::MatrixXd&)
bool MeanEuclidian(const std::vector<Eigen::MatrixXd>& covs, Eigen::MatrixXd& mean);

/// <summary>	Compute the Log Euclidiean Mean.\n
/// \f[ C_{\mu_\text{lE}} =\exp\left(\frac{1}{N} \sum_i{\log\left(C_i\right)}\right)\f]
/// </summary>
/// \copydetails MeanRiemann(const std::vector<Eigen::MatrixXd>&, Eigen::MatrixXd&)
bool MeanLogEuclidian(const std::vector<Eigen::MatrixXd>& covs, Eigen::MatrixXd& mean);

/// <summary>	Compute the Log Determinant Mean.\n
/// -# Compute the Classical Mean \f$ C_{\mu_\text{E}} \f$ (see <see cref="MeanEuclidian"/>)
/// -# Update with an iterative procedure that stops after 50 iterations or when criterion is under \f$ 10^{-4}\f$
///
/// \f[ C_{\mu_\text{lD}} = C_{\mu_\text{E}}\f]
/// Iterative process with \f$J\f$ while \f$ \text{iteration} < 50 \f$ and \f$ 10^{-4} < \left\lVert J-C_\mu \right\rVert \f$
/// \f[ \begin{aligned}
///		J &= \left(\frac{1}{N} \sum_i \left( 0.5 \times\left(C_{\mu_\text{lD}} + C_i \right)\right)^{-1} \right)^{-1}\\
///		C_{\mu_\text{lD}} &= J
///	\end{aligned}\f]
/// </summary>
/// \copydetails MeanRiemann(const std::vector<Eigen::MatrixXd>&, Eigen::MatrixXd&)
bool MeanLogDet(const std::vector<Eigen::MatrixXd>& covs, Eigen::MatrixXd& mean);

/// <summary>	Compute the Mean with the Kullback Mean.\n
/// The mean is the Geodesic center between the Euclidian and the Harmonic Mean.\n
/// \f[ C_{\mu_\text{K}} = \gamma \left( C_{\mu_{\text{E}}}, C_{\mu_{\text{H}}} \right) \f]
/// </summary>
/// \copydetails MeanRiemann(const std::vector<Eigen::MatrixXd>&, Eigen::MatrixXd&)
bool MeanKullback(const std::vector<Eigen::MatrixXd>& covs, Eigen::MatrixXd& mean);

/// <summary>	Compute the Wasserstein Mean.\n
/// -# Compute the Classical Mean \f$ C_{\mu_\text{E}} \f$ (see <see cref="MeanEuclidian"/>)
/// -# Update with an iterative procedure that stops after 50 iterations or when criterion is under \f$ 10^{-4}\f$
///
/// \f[ C_{\mu_\text{W}} = C_{\mu_{\text{E}}}\f]
/// Iterative process with \f$J\f$ while \f$ \text{iteration} < 50 \f$ and \f$ 10^{-4} < \left\lVert J-J_{-1} \right\rVert \f$
/// \f[ \begin{aligned}
///		J &= C_{\mu_\text{W}}^{1/2}\\
///		J &= \left(\frac{1}{N} \sum_i \left( J C_i J \right)^{1/2} \right)^{1/2}\\
///	\end{aligned}\f]
///	After the Iterative process : \f$ C_{\mu_\text{W}} = J*J \f$
/// </summary>
/// \copydetails MeanRiemann(const std::vector<Eigen::MatrixXd>&, Eigen::MatrixXd&)
/// \todo Doesn't work so good (after \f$10^{-3}\f$ precision with the pyriemann library).
bool MeanWasserstein(const std::vector<Eigen::MatrixXd>& covs, Eigen::MatrixXd& mean);

/// <summary>	Compute the Approximate joint diagonalization based log-Euclidean (ALE) Mean. \n
/// -# Compute the Approximate Joint Diagonalization \f$ C_\text{AJD} \f$ (see <see cref="AJDPham"/>)
/// -# Update with an iterative procedure that stops after 50 iterations or when criterion is under \f$ 10^{-4}\f$
///
/// \f[ C_{\mu_\text{ALE}} = C_\text{AJD}\f]
/// Iterative process with \f$J\f$ (and \f$U = \operatorname{diag}(\operatorname{diag}(\exp(J))\f$) while \f$ \text{iteration} < 50 \f$ and \f$ 10^{-4} < d_\text{R}(I_N,U) \f$
/// \f[ \begin{aligned}
///		J &= \frac{1}{N} \log\left(\sum_i \left( C_{\mu_\text{ALE}}^{\mathsf{T}} C_i C_{\mu_\text{ALE}} \right) \right)\\
///		U &= \operatorname{diag}(\operatorname{diag}(\exp(J))\\
///		C_{\mu_\text{ALE}} &= C_{\mu_\text{ALE}} * U^{-1/2}\\
///	\end{aligned}\f]
///	After the Iterative process : 
/// \f[ \begin{aligned}	
///		J &= \frac{1}{N} \log\left(\sum_i \left( C_{\mu_\text{ALE}}^{\mathsf{T}} C_i C_{\mu_\text{ALE}} \right) \right)\\
///		C_{\mu_\text{ALE}} &= \left(C_{\mu_\text{ALE}}^{-1}\right)^{\mathsf{T}} ~ \exp(J) ~ C_{\mu_\text{ALE}}^{-1}
///	\end{aligned}\f]
/// </summary>
/// \copydetails MeanRiemann(const std::vector<Eigen::MatrixXd>&, Eigen::MatrixXd&)
/// \todo Doesn't work => Need to implement <see cref="AJDPham"/> and check if it works next.
bool MeanALE(const std::vector<Eigen::MatrixXd>& covs, Eigen::MatrixXd& mean);

/// <summary>	Compute the Harmonic Mean.\n
/// \f[ C_{\mu_\text{H}} = (\frac{1}{N} \sum_i{C_i}^{-1})^{-1} \f]
/// </summary>
/// \copydetails MeanRiemann(const std::vector<Eigen::MatrixXd>&, Eigen::MatrixXd&)
bool MeanHarmonic(const std::vector<Eigen::MatrixXd>& covs, Eigen::MatrixXd& mean);

/// <summary>	Give the Identity Matrix.\n
///	\f[ C_{\mu_\text{I}} = I_N \f]
///	</summary>
/// \copydetails MeanRiemann(const std::vector<Eigen::MatrixXd>&, Eigen::MatrixXd&)
bool MeanIdentity(const std::vector<Eigen::MatrixXd>& covs, Eigen::MatrixXd& mean);

/// <summary>	Compute the Euclidian Mean of each class. </summary>
/// <param name="datasets">	The datasets one class by row and trials on colums (each sample is a feature vector). </param>
/// <param name="mean">	The mean. </param>
/// <returns>	True if it succeeds, false if it fails. </returns>
/// <remarks> Used for LDA Solver as in scikit-learn library (<a href="https://github.com/scikit-learn/scikit-learn/blob/1495f69242646d239d89a5713982946b8ffcf9d9/sklearn/discriminant_analysis.py#L72">scikit-learn Class Mean</a>).</remarks>
bool MeanClass(const std::vector<std::vector<Eigen::RowVectorXd>>& datasets, Eigen::MatrixXd& mean);