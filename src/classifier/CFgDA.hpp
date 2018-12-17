///-------------------------------------------------------------------------------------------------
/// 
/// \file CFgDA.hpp
/// 
/// \brief Fisher Geodesic Discriminant analysis Class.
/// 
/// \author Thibaut Monseigne (Inria).
/// 
/// \version 1.0.
/// 
/// \date 11/12/2018.
/// 
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// 
/// \remarks 
/// - Inspired by the work of Alexandre Barachant : <a href="https://github.com/alexandrebarachant/pyRiemann">pyRiemann</a> (<a href="https://github.com/alexandrebarachant/pyRiemann/blob/master/LICENSE">License</a>).
/// - <a href="https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis">Linear Discriminant analysis</a> with LSQR method inspired by <a href="http://scikit-learn.org">sklearn</a> (<a href="https://github.com/scikit-learn/scikit-learn/blob/master/COPYING">License</a>).
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include <ostream>
#include <vector>
#include "utils/Metrics.hpp"
#include "CFeatureClassifierLDA.hpp"

///----------------------------------------------------------------------------------------------------
///
/// <summary>	Fisher Geodesic Discriminant analysis Class. for FgMDM Class. </summary>
///
/// <seealso cref="CMatrixClassifierFgMDM" />
///
///----------------------------------------------------------------------------------------------------
class CFgDA
{
public:
	EMetrics m_Metric = Metric_Riemann;
	Eigen::MatrixXd m_Ref;
	Eigen::RowVectorXd m_Weight;
	CFeatureClassifierLDA m_LDA;

	//***********************
	//***** Constructor *****
	//***********************
	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>Initializes a new instance of the <see cref="CFgDA"/> class.</summary>
	///
	///----------------------------------------------------------------------------------------------------
	CFgDA() = default;

	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Finalizes an instance of the <see cref="CFgDA"/> class. </summary>
	///
	///----------------------------------------------------------------------------------------------------
	~CFgDA() = default;

	//********************
	//***** Computes *****
	//********************	
	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Computes the weight of FgDA. </summary>
	///
	/// <param name="dataset">	The dataset. </param>
	///
	/// <returns>	True if it succeeds, false if it fails. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	bool compute(const std::vector<Eigen::MatrixXd>& dataset);

	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Least squares solution method to compute weight. </summary>
	///
	/// <param name="dataset">	The dataset. </param>
	///
	/// <returns>	True if it succeeds, false if it fails. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	static bool lsqr(const std::vector<Eigen::RowVectorXd>& dataset);
	
	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Apply the filter to the sample. </summary>
	///
	/// <param name="in">	The sample. </param>
	/// <param name="out">	The filtered sample. </param>
	///
	/// <returns>	True if it succeeds, false if it fails. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	bool filter(const Eigen::MatrixXd& in, Eigen::MatrixXd& out) const;
	
	//*****************************
	//***** Override Operator *****
	//*****************************
	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Override the egal operator (with 1e-6 precision). </summary>
	///
	/// <param name="obj">	The second object. </param>
	///
	/// <returns>	True if the two <see cref="CFgDA"/> are equals. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	bool operator==(const CFgDA& obj) const;

	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Override the not egal operator (with 1e-6 precision). </summary>
	///
	/// <param name="obj">	The second object. </param>
	///
	/// <returns>	True if the two <see cref="CFgDA"/> are diffrents. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	bool operator!=(const CFgDA& obj) const;

	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Get the Classifier information for output. </summary>
	///
	/// <returns>	The Classifier print in stringstream. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	std::stringstream print() const;

	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Override the ostream operator. </summary>
	///
	/// <param name="os">	The ostream. </param>
	/// <param name="obj">	The object. </param>
	///
	/// <returns>	Return the modified ostream. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	friend std::ostream& operator <<(std::ostream& os, const CFgDA& obj);
};

