///-------------------------------------------------------------------------------------------------
/// 
/// \file CRebias.hpp
/// \brief Class used to add Rebias to Other Classifier.
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 27/08/2019.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>
#include <vector>
#include "utils/Metrics.hpp"
#include "3rd-party/tinyxml2.h"

/// <summary>	Rebias class. </summary>
class CRebias
{
public:
	/// <summary> Initializes a new instance of the <see cref="CRebias"/> class. </summary>
	CRebias() = default;
	/// <summary> Finalizes an instance of the <see cref="CRebias"/> class. </summary>
	~CRebias() = default;

	/// <summary> Computes the rebias matrix and reset the number of classification. </summary>
	/// <param name="datasets">	The data set one class by row and trials on colums. </param>
	/// <param name="metric">The metric.</param>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	bool computeRebias(const std::vector<std::vector<Eigen::MatrixXd>>& datasets, const EMetrics metric = Metric_Riemann);

	/// <summary> Applies the rebias on 2D vector of Matrix. </summary>
	/// <param name="in">The input 2D vector of matrix.</param>
	/// <param name="out">The output 2D vector of matrix.</param>
	void applyRebias(const std::vector<std::vector<Eigen::MatrixXd>>& in, std::vector<std::vector<Eigen::MatrixXd>>& out);
	/// <summary> Applies the rebias on vector of Matrix. </summary>
	/// <param name="in">The input vector of matrix.</param>
	/// <param name="out">The output vector of matrix.</param>
	void applyRebias(const std::vector<Eigen::MatrixXd>& in, std::vector<Eigen::MatrixXd>& out);
	/// <summary> Applies the rebias on Matrix. </summary>
	/// <param name="in">The input matrix.</param>
	/// <param name="out">The output matrix.</param>
	void applyRebias(const Eigen::MatrixXd& in, Eigen::MatrixXd& out);

	/// <summary> Updates the rebias. </summary>
	/// <param name="sample">The sample.</param>
	/// <param name="metric">The metric.</param>
	void updateRebias(const Eigen::MatrixXd& sample, const EMetrics metric = Metric_Riemann);

	/// <summary> Gets the bias. </summary>
	/// <returns></returns>
	Eigen::MatrixXd getBias() const { return m_bias; }
	
	/// <summary> Set the bias and the inverse square root of biais. </summary>
	/// <param name="bias">	The new bias.</param>
	void setBias(Eigen::MatrixXd& bias);

	//***********************
	//***** XML Manager *****
	//***********************
	/// <summary>	Save informations in xml element (Rebias and number of classification). </summary>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	bool save(tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* data) const;

	/// <summary>	Load informations in xml element (Rebias and number of classification). </summary>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	bool load(tinyxml2::XMLElement* data);

	//*****************************
	//***** Override Operator *****
	//*****************************
	/// \copydoc IMatrixClassifier::isEqual(const IMatrixClassifier&, const double) const
	bool isEqual(const CRebias& obj, const double precision = 1e-6) const;

	/// <summary>	Copy object value. </summary>
	/// <param name="obj">	The object to copy. </param>
	void copy(const CRebias& obj);

	/// <summary>	Get the Classifier information for output. </summary>
	/// <returns>	The Classifier print in stringstream. </returns>
	std::stringstream print() const;

	/// <summary>	Override the affectation operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	The copied object. </returns>
	CRebias& operator=(const CRebias& obj)
	{
		copy(obj);
		return *this;
	}

	/// <summary>	Override the egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	True if the two <see cref="CRebias"/> are equals. </returns>
	bool operator==(const CRebias& obj) const { return isEqual(obj); }

	/// <summary>	Override the not egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	True if the two <see cref="CRebias"/> are diffrents. </returns>
	bool operator!=(const CRebias& obj) const { return !isEqual(obj); }

	/// <summary>	Override the ostream operator. </summary>
	/// <param name="os">	The ostream. </param>
	/// <param name="obj">	The object. </param>
	/// <returns>	Return the modified ostream. </returns>
	friend std::ostream& operator <<(std::ostream& os, const CRebias& obj)
	{
		os << obj.print().str();
		return os;
	}

	//*********************
	//***** Variables *****
	//*********************
	/// <summary>	Number of classification launched. </summary>
	size_t m_NClassify = 0;

protected:
	/// <summary>	Rebias Matrix. </summary>
	Eigen::MatrixXd m_bias, m_biasIS;

};
