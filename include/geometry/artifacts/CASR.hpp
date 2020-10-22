///-------------------------------------------------------------------------------------------------
/// 
/// \file CASR.hpp
/// \brief Class used to use Artifact Subspace Reconstruction Algorithm.
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 27/08/2020.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// 
///-------------------------------------------------------------------------------------------------
#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>
#include "3rd-party/tinyxml2.h"
#include "utils/Metrics.hpp"
#include "utils/Misc.hpp"


class CASR
{
public:

	CASR() = default;	///< Initializes a new instance of the <see cref="CASR"/> class.

	/// <summary> Initializes a new instance of the <see cref="CASR"/> class with specified metric </summary>
	/// <remarks> Only Euclidian and Riemmann metrics are implemnted If other is selected, Euclidian is used. </remarks>
	//explicit CASR(const EMetric metric) : m_metric(metric) {}

	~CASR() = default;	///< Finalizes an instance of the <see cref="CASR"/> class.

	/// <summary>	Trains the specified dataset. </summary>
	/// <param name="dataset">	The dataset. </param>
	/// <param name="rejectionLimit">	The rejection limit. </param>
	/// <returns>	<c>True</c> if it succeeds, <c>false</c> if it fails. </returns>
	bool train(const std::vector<Eigen::MatrixXd>& dataset, const double rejectionLimit = 5);

	/// <summary>	Apply the ASR algorithm to the input signal. </summary>
	/// <param name="in">	The input signal. </param>
	/// <param name="out">	The corrected signal. </param>
	/// <returns>	<c>True</c> if it succeeds, <c>false</c> if it fails. </returns>
	bool process(const Eigen::MatrixXd& in, Eigen::MatrixXd& out);


	/*

	static const std::map<size_t, std::pair<int, int>> Filters;

	void computeStat(const Eigen::MatrixXd& data);
	
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
*/
	//***************************
	//***** Getter / Setter *****
	//***************************

	/// <summary> Set the metric to use (only Riemann and euclidian is used. </summary>
	/// <param name="metric">The metric.</param>
	/// <remarks> If invalid metric is used Euclidian is selected. </remarks>
	void setMetric(EMetric& metric) { m_metric = (metric == EMetric::Riemann) ? EMetric::Riemann : EMetric::Euclidian; }

	/// <summary>	Gets the metric. </summary>
	/// <returns>	<c>EMetric</c>. </returns>
	EMetric getMetric() const { return m_metric; }

	/// <summary>	Gets the median matrix. </summary>
	/// <returns>	<c>Eigen::MatrixXd</c>. </returns>
	Eigen::MatrixXd getMedian() const { return m_median; }

	/// <summary>	Gets the transformation matrix. </summary>
	/// <returns>	<c>Eigen::MatrixXd</c>. </returns>
	Eigen::MatrixXd getTransformMatrix() const { return m_treshold; }

	//*****************************
	//***** Override Operator *****
	//*****************************
	bool isEqual(const CASR& obj, const double precision = 1e-6) const;

	/// <summary>	Copy object value. </summary>
	/// <param name="obj">	The object to copy. </param>
	void copy(const CASR& obj);

	/// <summary>	Get the ASR information for output. </summary>
	/// <returns>	The ASR print in stringstream. </returns>
	std::string toString() const;

	/// <summary>	Override the affectation operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	The copied object. </returns>
	CASR& operator=(const CASR& obj)
	{
		copy(obj);
		return *this;
	}

	/// <summary>	Override the egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	True if the two <see cref="CASR"/> are equals. </returns>
	bool operator==(const CASR& obj) const { return isEqual(obj); }

	/// <summary>	Override the not egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	True if the two <see cref="CASR"/> are diffrents. </returns>
	bool operator!=(const CASR& obj) const { return !isEqual(obj); }

	/// <summary>	Override the ostream operator. </summary>
	/// <param name="os">	The ostream. </param>
	/// <param name="obj">	The object. </param>
	/// <returns>	Return the modified ostream. </returns>
	friend std::ostream& operator <<(std::ostream& os, const CASR& obj)
	{
		os << obj.toString();
		return os;
	}

protected:

	//*********************
	//***** Variables *****
	//*********************
	EMetric m_metric    = EMetric::Euclidian;	///< Metric Used for computes (only euclidian and Riemann are implemented
	size_t m_nChannel   = 0;					///< Number of channel (dimension)
	bool m_trivial      = true;					///< Define if previous sample was trivial to reconstruct
	double m_maxChannel = 1;					///< Number of channel (dimension) to reconstruct in fraction, 0 for nothing 1 for all
	Eigen::MatrixXd m_median;					///< Median computed with train dataset
	Eigen::MatrixXd m_treshold;					///< Treshold matrix computed with train dataset
	Eigen::MatrixXd m_r;						///< Last Reconstruction matrix
	Eigen::MatrixXd m_cov;						///< Last Covariance matrix
};
