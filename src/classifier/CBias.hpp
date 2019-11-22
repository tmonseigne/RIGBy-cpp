///-------------------------------------------------------------------------------------------------
/// 
/// \file CBias.hpp
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

class CBias
{
public:
	/// <summary> Initializes a new instance of the <see cref="CBias"/> class. </summary>
	CBias() = default;
	/// <summary> Finalizes an instance of the <see cref="CBias"/> class. </summary>
	~CBias() = default;

	/// <summary> Computes the Bias matrix and reset the number of classification. </summary>
	/// <param name="datasets">	The data set one class by row and trials on colums. </param>
	/// <param name="metric">The metric.</param>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	bool computeBias(const std::vector<std::vector<Eigen::MatrixXd>>& datasets, const EMetrics metric = Metric_Riemann);
	
	/// <summary> Computes the Bias matrix and reset the number of classification. </summary>
	/// <param name="datasets">	The data set is a vector of trial. </param>
	/// <param name="metric">The metric.</param>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	bool computeBias(const std::vector<Eigen::MatrixXd>& datasets, const EMetrics metric = Metric_Riemann);

	/// <summary> Applies the Bias on 2D vector of Matrix. </summary>
	/// <param name="in">The input 2D vector of matrix.</param>
	/// <param name="out">The output 2D vector of matrix.</param>
	void applyBias(const std::vector<std::vector<Eigen::MatrixXd>>& in, std::vector<std::vector<Eigen::MatrixXd>>& out);
	/// <summary> Applies the Bias on vector of Matrix. </summary>
	/// <param name="in">The input vector of matrix.</param>
	/// <param name="out">The output vector of matrix.</param>
	void applyBias(const std::vector<Eigen::MatrixXd>& in, std::vector<Eigen::MatrixXd>& out);
	/// <summary> Applies the Bias on Matrix. </summary>
	/// <param name="in">The input matrix.</param>
	/// <param name="out">The output matrix.</param>
	void applyBias(const Eigen::MatrixXd& in, Eigen::MatrixXd& out);

	/// <summary> Updates the Bias. </summary>
	/// <param name="sample">The sample.</param>
	/// <param name="metric">The metric.</param>
	void updateBias(const Eigen::MatrixXd& sample, const EMetrics metric = Metric_Riemann);

	/// <summary> Gets the bias. </summary>
	/// <returns></returns>
	Eigen::MatrixXd getBias() const { return m_bias; }
	
	/// <summary> Set the bias and the inverse square root of biais. </summary>
	/// <param name="bias">	The new bias.</param>
	void setBias(const Eigen::MatrixXd& bias);

	//***********************
	//***** XML Manager *****
	//***********************
	/// <summary>	Saves the Bias information in an XML file. </summary>
	/// <param name="filename">	Filename. </param>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	bool saveXML(const std::string& filename) const;

	/// <summary>	Loads the Bias information from an XML file. </summary>
	/// <param name="filename">	Filename. </param>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	bool loadXML(const std::string& filename);

	/// <summary>	Save informations in xml element (Bias and number of classification). </summary>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	bool saveAdditional(tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* data) const;

	/// <summary>	Load informations in xml element (Bias and number of classification). </summary>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	bool loadAdditional(tinyxml2::XMLElement* data);

	//*****************************
	//***** Override Operator *****
	//*****************************
	/// \copydoc IMatrixClassifier::isEqual(const IMatrixClassifier&, const double) const
	bool isEqual(const CBias& obj, const double precision = 1e-6) const;

	/// <summary>	Copy object value. </summary>
	/// <param name="obj">	The object to copy. </param>
	void copy(const CBias& obj);

	/// <summary>	Get the Classifier information for output. </summary>
	/// <returns>	The Classifier print in stringstream. </returns>
	std::stringstream print() const;

	/// <summary>	Override the affectation operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	The copied object. </returns>
	CBias& operator=(const CBias& obj)
	{
		copy(obj);
		return *this;
	}

	/// <summary>	Override the egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	True if the two <see cref="CBias"/> are equals. </returns>
	bool operator==(const CBias& obj) const { return isEqual(obj); }

	/// <summary>	Override the not egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	True if the two <see cref="CBias"/> are diffrents. </returns>
	bool operator!=(const CBias& obj) const { return !isEqual(obj); }

	/// <summary>	Override the ostream operator. </summary>
	/// <param name="os">	The ostream. </param>
	/// <param name="obj">	The object. </param>
	/// <returns>	Return the modified ostream. </returns>
	friend std::ostream& operator <<(std::ostream& os, const CBias& obj)
	{
		os << obj.print().str();
		return os;
	}

	//*********************
	//***** Variables *****
	//*********************
	/// <summary>	Number of classification launched. </summary>
	size_t m_N = 0;

protected:
	/// <summary>	Bias Matrix. </summary>
	Eigen::MatrixXd m_bias, m_biasIS;
};
