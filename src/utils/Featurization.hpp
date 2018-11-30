///-------------------------------------------------------------------------------------------------
/// 
/// \file Featurization.hpp
/// 
/// \brief All functions to transform Covariance matrix to feature vector.
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

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Compute the features vector of covariance matrix with the selected method. </summary>
///
/// <param name="matrix">	 	The covariance matrix. </param>
/// <param name="ref">		 	The reference Matrix (usefull for Tangent Space Featurization). </param>
/// <param name="rowVector">	The Feature Vector. </param>
/// <param name="tangent">   	(Optional) True to use tangent space featurization, Upper Triangle Squeeze if false. </param>
///
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
///-------------------------------------------------------------------------------------------------
bool Featurization(const Eigen::MatrixXd& matrix, Eigen::RowVectorXd& rowVector, bool tangent = true, const Eigen::MatrixXd& ref = Eigen::MatrixXd());

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Squeeze the upper triangle of NxN square matrix to a N(N+1)/2 Vector.
///
/// <table align="center" border="0">
/// <tr><th>Upper Triangle Matrix</th><th></th><th>Row Major Upper Triangle Squeeze</th> <th></th> <th>Diagonal Major Upper Triangle Squeeze</th></tr>
/// <tr><td>\f[ \begin{pmatrix} a&b&c\\d&e&f\\g&h&i \end{pmatrix} \Rightarrow \begin{pmatrix} a&b&c\\0&e&f\\0&0&i \end{pmatrix} \f]</td>
/// 	<td><pre>		</pre></td>
/// 	<td>\f[ \begin{pmatrix} a&b&c\\d&e&f\\g&h&i \end{pmatrix} \Rightarrow \begin{pmatrix} a&b&c&e&f&i \end{pmatrix} \f]</td>
/// 	<td><pre>		</pre></td>
/// 	<td>\f[\begin{pmatrix} a&b&c\\d&e&f\\g&h&i \end{pmatrix} \Rightarrow \begin{pmatrix} a&e&i&b&f&c \end{pmatrix} \f]</td></tr>
/// </table>
/// </summary>
/// 
/// <param name="matrix">		The NXN matrix. </param>
/// <param name="rowVector">   The N(N+1)/2 vector. </param>
/// <param name="rowMajor">	Get the values row by row if true,  diagonal by diagonal if false. </param>
///
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
///-------------------------------------------------------------------------------------------------
bool SqueezeUpperTriangle(const Eigen::MatrixXd& matrix, Eigen::RowVectorXd& rowVector, bool rowMajor = true);

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Compute the upper triangle of N(N+1)/2 Vector to a NxN square matrix.
///
/// <table align="center" border="0">
/// <tr><th>Row Major Method</th> <th></th> <th>Diagonal Major Method</th></tr>
/// <tr><td>\f[ \begin{pmatrix} a&b&c&d&e&f \end{pmatrix} \Rightarrow \begin{pmatrix} a&b&c\\0&d&e\\0&0&f \end{pmatrix} \f]</td>
/// 	<td><pre>		</pre></td>
/// 	<td>\f[ \begin{pmatrix} a&b&c&d&e&f \end{pmatrix} \Rightarrow \begin{pmatrix} a&d&f\\0&b&e\\0&0&c \end{pmatrix}  \f]</td></tr>
/// </table>
/// </summary>
/// 
/// <param name="rowVector">   The N(N+1)/2 vector. </param>
/// <param name="matrix">		The NXN matrix. </param>
/// <param name="rowMajor">	Get the values row by row if true,  diagonal by diagonal if false. </param>
///
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
///-------------------------------------------------------------------------------------------------
bool Vector2UpperTriangle(const Eigen::RowVectorXd& rowVector, Eigen::MatrixXd& matrix, bool rowMajor = true);

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Project a covariance matrices in the tangent space according to the given reference point. 
/// 	
/// \f[ 
/// \begin{aligned}
///		J &= \log{\left(M_\text{Ref}^{-1/2} ~  M ~  M_\text{Ref}^{-1/2}\right)} \\
///		M_\text{Coeffs} &= \begin{pmatrix} 
///								1 & \sqrt{2} & \cdots & \sqrt{2} \\
///								0 & 1 & \ddots & \sqrt{2} \\
///								\vdots & \ddots & \ddots & \vdots\\
///								0 & \cdots & \cdots & 1 
///							\end{pmatrix}
///	\end{aligned} \\
///	\text{With : } V_J = \operatorname{SqueezeUpperTriangle}(J) \quad \text{ and } \quad V_\text{Coeffs} = \operatorname{SqueezeUpperTriangle}(M_\text{Coeffs})\\
///	\Rightarrow V_\text{Ts} = V_J \odot V_\text{Coeffs}
///	\f]
/// </summary>
///
/// <param name="matrix">		The NXN covariance matrix. </param>
/// <param name="rowVector">   The N(N+1)/2 row. </param>
/// <param name="ref">   		(Optional) The NXN reference matrix (use the identity Matrix if empty). </param>
///
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
///-------------------------------------------------------------------------------------------------
bool TangentSpace(const Eigen::MatrixXd& matrix, Eigen::RowVectorXd& rowVector, const Eigen::MatrixXd& ref = Eigen::MatrixXd());

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Project a Tangent space vectors in the manifold according to the given reference point. 
///
/// \f[
/// \begin{aligned}
///	\text{With : } M_\text{Ts} &= \operatorname{Vector2UpperTriangle}(V_\text{Ts}) \quad \text{ and } \quad \mathsf{U}_{M}\text{ the upper triangular matrix.}\\
///	M_\text{Coeffs} &= \operatorname{diag}\left(M_\text{Ts}\right) + \frac{\mathsf{U}_{M_\text{Ts}} + \mathsf{U}_{M_\text{Ts}}^{\mathsf{T}}}{\sqrt{2}}\\
///	\Rightarrow M &= M_\text{Ref}^{1/2} ~  \exp{\left(M_\text{Coeffs}\right)} ~  M_\text{Ref}^{1/2}
///	\end{aligned}
///	\f]
/// </summary>
///
/// <param name="rowVector">   The N(N+1)/2 row. </param>
/// <param name="matrix">	The NXN covariance matrix. </param>
/// <param name="ref">   	(Optional) The NXN reference matrix (use the identity Matrix if empty). </param>
///
/// <returns>	True if it succeeds, false if it fails. </returns>
/// 
///-------------------------------------------------------------------------------------------------
bool UnTangentSpace(const Eigen::RowVectorXd& rowVector, Eigen::MatrixXd& matrix, const Eigen::MatrixXd& ref = Eigen::MatrixXd());
