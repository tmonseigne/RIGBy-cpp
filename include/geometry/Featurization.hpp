///-------------------------------------------------------------------------------------------------
/// 
/// \file Featurization.hpp
/// \brief All functions to transform Covariance matrix to feature vector.
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 26/10/2018.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>

namespace Geometry {

/// <summary>	Compute the features vector of covariance matrix with the selected method. </summary>
/// <param name="in">	 	The covariance in. </param>
/// <param name="out">		The Feature Vector. </param>
/// <param name="tangent">  (Optional) True to use tangent space featurization, Upper Triangle Squeeze if false. </param>
/// <param name="ref">		The reference Matrix (usefull for Tangent Space Featurization). </param>
/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
bool Featurization(const Eigen::MatrixXd& in, Eigen::RowVectorXd& out, bool tangent = true, const Eigen::MatrixXd& ref = Eigen::MatrixXd());

/// <summary>	Compute the covariance matrix of features vector with the selected method. </summary>
/// <param name="in">		The Feature Vector. </param>
/// <param name="out">	 	The covariance out. </param>
/// <param name="tangent">	(Optional) True to use tangent space featurization, Upper Triangle Squeeze if false. </param>
/// <param name="ref">		The reference Matrix (usefull for Tangent Space Featurization). </param>
/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
bool UnFeaturization(const Eigen::RowVectorXd& in, Eigen::MatrixXd& out, bool tangent = true, const Eigen::MatrixXd& ref = Eigen::MatrixXd());

/// <summary>	Squeeze the upper triangle of \f$N \times N\f$ square matrix to a \f$\frac{N\left(N+1\right)}{2}\f$ Vector.
/// <table align="center" border="0">
/// <tr><th>Upper Triangle Matrix</th><th></th><th>Row Major Upper Triangle Squeeze</th> <th></th> <th>Diagonal Major Upper Triangle Squeeze</th></tr>
/// <tr><td>\f[ \begin{pmatrix} a&b&c\\d&e&f\\g&h&i \end{pmatrix} \Rightarrow \begin{pmatrix} a&b&c\\0&e&f\\0&0&i \end{pmatrix} \f]</td>
/// 	<td><pre>		</pre></td>
/// 	<td>\f[ \begin{pmatrix} a&b&c\\d&e&f\\g&h&i \end{pmatrix} \Rightarrow \begin{pmatrix} a&b&c&e&f&i \end{pmatrix} \f]</td>
/// 	<td><pre>		</pre></td>
/// 	<td>\f[\begin{pmatrix} a&b&c\\d&e&f\\g&h&i \end{pmatrix} \Rightarrow \begin{pmatrix} a&e&i&b&f&c \end{pmatrix} \f]</td></tr>
/// </table>
/// </summary>
/// <param name="in">		The \f$N \times N\f$ square matrix. </param>
/// <param name="out">		The \f$\frac{N\left(N+1\right)}{2}\f$ Vector. </param>
/// <param name="rowMajor">	Get the values row by row if true,  diagonal by diagonal if false. </param>
/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
bool SqueezeUpperTriangle(const Eigen::MatrixXd& in, Eigen::RowVectorXd& out, bool rowMajor = true);

/// <summary>	Compute the upper triangle of \f$\frac{N\left(N+1\right)}{2}\f$ Vector to a \f$N \times N\f$ square matrix.
/// <table align="center" border="0">
/// <tr><th>Row Major Method</th> <th></th> <th>Diagonal Major Method</th></tr>
/// <tr><td>\f[ \begin{pmatrix} a&b&c&d&e&f \end{pmatrix} \Rightarrow \begin{pmatrix} a&b&c\\0&d&e\\0&0&f \end{pmatrix} \f]</td>
/// 	<td><pre>		</pre></td>
/// 	<td>\f[ \begin{pmatrix} a&b&c&d&e&f \end{pmatrix} \Rightarrow \begin{pmatrix} a&d&f\\0&b&e\\0&0&c \end{pmatrix}  \f]</td></tr>
/// </table>
/// </summary>
/// <param name="in">		The \f$\frac{N\left(N+1\right)}{2}\f$ Vector. </param>
/// <param name="out">		The \f$N \times N\f$ square matrix. </param>
/// <param name="rowMajor">	Get the values row by row if true,  diagonal by diagonal if false. </param>
/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
bool UnSqueezeUpperTriangle(const Eigen::RowVectorXd& in, Eigen::MatrixXd& out, bool rowMajor = true);

/// <summary>	Project a covariance matrices (\f$M\f$) in the tangent space (\f$\mathcal{T}\f$) according to the given reference point (\f$M_\text{Ref}\f$). <br/>
///
/// - Compute the transformation matrix for the covariance matrix \f$M\f$ with the reference matrix \f$M_\text{Ref}\f$ and squeeze ths matrix (see <see cref="SqueezeUpperTriangle"/>).
/// \f[
/// \begin{aligned}
/// J &= \log{\left(M_\text{Ref}^{-1/2} \times  M \times  M_\text{Ref}^{-1/2}\right)}\\
/// V_J &= \operatorname{SqueezeUpperTriangle}(J)
///	\end{aligned}
///	\f]
/// - Compute a coefficient Vector to apply to transformation vector.
/// \f[ 
/// \begin{aligned}
///		M_\text{Coeffs} &= \begin{pmatrix} 
///								1 & \sqrt{2} & \cdots & \sqrt{2} \\
///								0 & 1 & \ddots & \sqrt{2} \\
///								\vdots & \ddots & \ddots & \vdots\\
///								0 & \cdots & \cdots & 1 
///							\end{pmatrix}\\
///	V_\text{Coeffs} &= \operatorname{SqueezeUpperTriangle}(M_\text{Coeffs})\\
///	\end{aligned}
///	\f]
///	- Compute the element wise product of the two vectors to have the tangent space Projection \f$\zeta_M\f$
/// \f[ \zeta_M = V_J \odot V_\text{Coeffs} \f]
/// </summary>
/// <param name="in">	The \f$N \times N\f$ covariance matrix. </param>
/// <param name="out">	The  \f$\frac{N\left(N+1\right)}{2}\f$ row. </param>
/// <param name="ref">  (Optional) The \f$N \times N\f$ reference in (use the identity Matrix if empty). </param>
/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
bool TangentSpace(const Eigen::MatrixXd& in, Eigen::RowVectorXd& out, const Eigen::MatrixXd& ref = Eigen::MatrixXd());

/// <summary>	Project a Tangent space vectors in the manifold according to the given reference point.  <br/>
/// \f[
/// \begin{aligned}
///	\text{With : } M_\text{Ts} &= \operatorname{UnSqueezeUpperTriangle}(V_\text{Ts}) \quad \text{ and } \quad \mathsf{U}_{M}\text{ the upper triangular out.}\\
///	M_\text{Coeffs} &= \operatorname{diag}\left(M_\text{Ts}\right) + \frac{\mathsf{U}_{M_\text{Ts}} + \mathsf{U}_{M_\text{Ts}}^{\mathsf{T}}}{\sqrt{2}}\\
///	\Rightarrow M &= M_\text{Ref}^{1/2} ~  \exp{\left(M_\text{Coeffs}\right)} ~  M_\text{Ref}^{1/2}
///	\end{aligned}
///	\f]
/// </summary>
/// <param name="in">   The  \f$\frac{N\left(N+1\right)}{2}\f$ row. </param>
/// <param name="out">	The \f$N \times N\f$ covariance matrix. </param>
/// <param name="ref">	(Optional) The \f$N \times N\f$ reference out (use the identity Matrix if empty). </param>
/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
bool UnTangentSpace(const Eigen::RowVectorXd& in, Eigen::MatrixXd& out, const Eigen::MatrixXd& ref = Eigen::MatrixXd());

}  // namespace Geometry
