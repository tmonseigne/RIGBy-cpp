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

#include "geometry/Basics.hpp"
#include "geometry/3rd-party/tinyxml2.h"
#include "geometry/Metrics.hpp"
#include "geometry/Misc.hpp"

namespace Geometry {

/// <summary> Class For Artifact Subspace Reconstruction (ASR) Algorithm. </summary>
class CASR
{
public:

	CASR() = default;	///< Initializes a new instance of the <see cref="CASR"/> class.

	/// <summary> Initializes a new instance of the <see cref="CASR"/> class with specified <c>metric</c>. </summary>
	/// <remarks> Only Euclidian and Riemmann metrics are implemented If other is selected, Euclidian is used. </remarks>
	explicit CASR(const EMetric& metric) { setMetric(metric); }

	/// <summary> Initializes a new instance of the <see cref="CASR"/> class with specified <c>metric</c> and train with the specified <c>dataset</c>. </summary>
	/// <remarks> Only Euclidian and Riemmann metrics are implemented If other is selected, Euclidian is used. </remarks>
	explicit CASR(const EMetric& metric, const std::vector<Eigen::MatrixXd>& dataset)
	{
		setMetric(metric);
		train(dataset);
	}

	~CASR() = default;	///< Finalizes an instance of the <see cref="CASR"/> class.

	/// <summary>	Trains the specified dataset. </summary>
	/// <param name="dataset">	The dataset. </param>
	/// <param name="rejectionLimit">	The rejection limit. </param>
	/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
	bool train(const std::vector<Eigen::MatrixXd>& dataset, const double rejectionLimit = 5);

	/// <summary>	Apply the ASR algorithm to the input signal. </summary>
	/// <param name="in">	The input signal. </param>
	/// <param name="out">	The corrected signal. </param>
	/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
	bool process(const Eigen::MatrixXd& in, Eigen::MatrixXd& out);

	//***************************
	//***** Getter / Setter *****
	//***************************

	/// <summary> Set the metric to use (only Riemann and euclidian is used. </summary>
	/// <param name="metric">The metric. </param>
	/// <remarks> If invalid metric is used Euclidian is selected. </remarks>
	void setMetric(const EMetric& metric) { m_metric = (metric == EMetric::Riemann) ? EMetric::Riemann : EMetric::Euclidian; }

	/// <summary> Sets the number of channel (dimension) to reconstruct in fraction, 0 for nothing 1 for all. </summary>
	/// <param name="max">	The maximum ratio. </param>
	/// <remarks>	If value isn't in [0;1], this function does nothing. </remarks>
	void setMaxChannel(const double max) { if (InRange(max, 0.0, 1.0)) { m_maxChannel = max; } }

	/// <summary> Sets the differents matrices : median matrix, trheshold matrix, reconstruction matrix and covariance matrix. </summary>
	/// <param name="median">		The median matrix. </param>
	/// <param name="threshold">	The threshold matrix. </param>
	/// <param name="reconstruct">	(Optional) The reconstruct matrix. </param>
	/// <param name="covariance">	(Optional) The covariance matrix. </param>
	/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
	/// <remarks>	All matrices must be square with same size (or empty for reconstruct and covariance matrix).
	/// Trivial trigger is set to true. </remarks>
	bool setMatrices(const Eigen::MatrixXd& median, const Eigen::MatrixXd& threshold,
					 const Eigen::MatrixXd& reconstruct = Eigen::MatrixXd(), const Eigen::MatrixXd& covariance = Eigen::MatrixXd());

	EMetric getMetric() const { return m_metric; }						///< Get the metric.
	size_t getChannelNumber() const { return m_nChannel; }				///< Get the matrices number of channel.
	double getMaxChannel() const { return m_maxChannel; }				///< Get the number of channel (dimension) to reconstruct in fraction.
	Eigen::MatrixXd getMedian() const { return m_median; }				///< Get the median matrix.
	Eigen::MatrixXd getThresholdMatrix() const { return m_threshold; }	///< Get the threshold matrix.

	//***********************
	//***** XML Manager *****
	//***********************
	/// <summary>	Saves the ASR information in an XML file. </summary>
	/// <param name="filename">	Filename. </param>
	/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
	bool saveXML(const std::string& filename) const;

	/// <summary>	Loads the ASR information from an XML file. </summary>
	/// <param name="filename">	Filename. </param>
	/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
	bool loadXML(const std::string& filename);

	//*****************************
	//***** Override Operator *****
	//*****************************
	/// <summary>	Check if object are equals (with a precision tolerance). </summary>
	/// <param name="obj">			The second object. </param>
	/// <param name="precision">	Precision for matrix comparison. </param>
	/// <returns>	<c>True</c> if the two elements are equals (with a precision tolerance), <c>False</c> otherwise. </returns>
	bool isEqual(const CASR& obj, const double precision = 1e-6) const;

	/// <summary>	Copy object value. </summary>
	/// <param name="obj">	The object to copy. </param>
	void copy(const CASR& obj);

	/// <summary>	Get the ASR information for output. </summary>
	/// <returns>	The ASR print in stringstream. </returns>
	std::stringstream print() const;

	/// <summary>	Override the affectation operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	The copied object. </returns>
	CASR& operator=(const CASR& obj)
	{
		copy(obj);
		return *this;
	}

	/// <summary>	Override the equal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	<c>True</c> if the two <see cref="CASR"/> are equals. </returns>
	bool operator==(const CASR& obj) const { return isEqual(obj); }

	/// <summary>	Override the not equal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	<c>True</c> if the two <see cref="CASR"/> are diffrents. </returns>
	bool operator!=(const CASR& obj) const { return !isEqual(obj); }

	/// <summary>	Override the ostream operator. </summary>
	/// <param name="os">	The ostream. </param>
	/// <param name="obj">	The object. </param>
	/// <returns>	Return the modified ostream. </returns>
	friend std::ostream& operator <<(std::ostream& os, const CASR& obj)
	{
		os << obj.print().str();
		return os;
	}

protected:

	//*********************
	//***** Variables *****
	//*********************
	EMetric m_metric    = EMetric::Euclidian;	///< Metric Used for computes (only euclidian and Riemann are implemented
	size_t m_nChannel   = 0;					///< Number of channel (dimension)
	double m_maxChannel = 1;					///< Number of channel (dimension) to reconstruct in fraction, 0 for nothing 1 for all
	bool m_trivial      = true;					///< Define if previous sample was trivial to reconstruct
	Eigen::MatrixXd m_median;					///< Median computed with train dataset
	Eigen::MatrixXd m_threshold;				///< Threshold matrix computed with train dataset
	Eigen::MatrixXd m_r;						///< Last Reconstruction matrix
	Eigen::MatrixXd m_cov;						///< Last Covariance matrix
};

}  // namespace Geometry
