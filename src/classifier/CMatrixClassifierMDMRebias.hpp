///-------------------------------------------------------------------------------------------------
/// 
/// \file CMatrixClassifierMDM.hpp
/// \brief Class of Minimum Distance to Mean (MDM) Classifier
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 10/12/2018.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include "CMatrixClassifierMDM.hpp"
#include "utils/Metrics.hpp"

/// <summary>	Class of Minimum Distance to Mean (MDM) Classifier. </summary>
/// <seealso cref="IMatrixClassifier" />
class CMatrixClassifierMDMRebias : public CMatrixClassifierMDM
{
public:
	//***********************	
	//***** Constructor *****
	//***********************	
	/// <summary>	Default constructor. Initializes a new instance of the <see cref="CMatrixClassifierMDMRebias"/> class. </summary>
	CMatrixClassifierMDMRebias() = default;

	/// <summary>	Default Copy constructor. Initializes a new instance of the <see cref="CMatrixClassifierMDMRebias"/> class. </summary>
	/// <param name="obj">	Initial object. </param>
	CMatrixClassifierMDMRebias(const CMatrixClassifierMDMRebias& obj) { *this = obj; }

	/// \copydoc CMatrixClassifierMDM(CMatrixClassifierMDM&&)
	CMatrixClassifierMDMRebias(CMatrixClassifierMDMRebias&& obj) = default;

	/// <summary>	Initializes a new instance of the <see cref="CMatrixClassifierMDMRebias"/> class and set base members. </summary>
	/// \copydetails CMatrixClassifierMDM(const size_t, const EMetrics)
	explicit CMatrixClassifierMDMRebias(const size_t nbClass, const EMetrics metric) : CMatrixClassifierMDM(nbClass, metric) { }

	/// <summary>	Finalizes an instance of the <see cref="CMatrixClassifierMDMRebias"/> class. </summary>
	virtual ~CMatrixClassifierMDMRebias() = default;

	//**********************
	//***** Classifier *****
	//**********************

	/// \copybrief IMatrixClassifier::train(const std::vector<std::vector<Eigen::MatrixXd>>&)
	/// <summary>	
	/// -# Compute the mean of all trials with the metric (<see cref="EMetrics" />) in <see cref="m_Metric"/> member as reference and store this in <see cref="m_Rebias"/> member.
	/// -# Set the good number of classes
	/// -# Apply an affine transformation on each trials with the reference : \f$ S_\text{new} = R^{-1/2} * S * {R^{-1/2}}^{\mathsf{T}} \f$
	/// -# Compute the mean of each class (row), on transformed trials, with the metric (<see cref="EMetrics" />) in <see cref="m_Metric"/> member.
	/// -# Set the number of trials for each class.
	///	</summary>
	/// \copydetails IMatrixClassifier::train(const std::vector<std::vector<Eigen::MatrixXd>>&)
	bool train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets) override;

	/// \copybrief CMatrixClassifierMDM::classify(const Eigen::MatrixXd&, size_t&, std::vector<double>&, std::vector<double>&, const EAdaptations, const size_t&)
	/// <summary>
	/// Apply an affine transformation on the trial (sample) with the reference : \f$ S_\text{new} = R^{-1/2} * S * {R^{-1/2}}^{\mathsf{T}} \f$ \n
	/// Update the reference with the current sample the first time and next with the Geodesic between the reference and the current sample.\n
	/// With \f$ \gamma_m \f$ the Geodesic (<see cref="Geodesic" />) with the metric \f$ m \f$ (<see cref="EMetrics" />) and \f$ N_c \f$ the number of classification : \f$ R = \gamma_\text{m}\left( R,S,\frac{1}{N_c} \right) \f$ \n
	///	</summary>
	/// \copydetails CMatrixClassifierMDM::classify(const Eigen::MatrixXd&, size_t&, std::vector<double>&, std::vector<double>&, const EAdaptations, const size_t&)
	bool classify(const Eigen::MatrixXd& sample, size_t& classId, std::vector<double>& distance, std::vector<double>& probability,
				  const EAdaptations adaptation = Adaptation_None, const size_t& realClassId = std::numeric_limits<std::size_t>::max()) override;

	//*****************************
	//***** Override Operator *****
	//*****************************

	/// \copydoc CMatrixClassifierMDM::isEqual(const CMatrixClassifierMDM&, const double) const
	bool isEqual(const CMatrixClassifierMDMRebias& obj, const double precision = 1e-6) const;

	/// \copydoc CMatrixClassifierMDM::copy(const CMatrixClassifierMDM&)
	void copy(const CMatrixClassifierMDMRebias& obj);

	/// \copybrief CMatrixClassifierMDM::getType()
	/// <returns>	Minimum Distance to Mean. </returns>
	std::string getType() const override { return "Minimum Distance to Mean REBIAS"; }

	/// <summary>	Override the affectation operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	The copied object. </returns>
	CMatrixClassifierMDMRebias& operator=(const CMatrixClassifierMDMRebias& obj)
	{
		copy(obj);		
		return *this;
	}

	/// <summary>	Don't Override the move operator. </summary>
	CMatrixClassifierMDMRebias& operator=(CMatrixClassifierMDMRebias&& obj) = default;

	/// <summary>	Override the egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	True if the two <see cref="CMatrixClassifierMDMRebias"/> are equals. </returns>
	bool operator==(const CMatrixClassifierMDMRebias& obj) const { return isEqual(obj); }

	/// <summary>	Override the not egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	True if the two <see cref="CMatrixClassifierMDMRebias"/> are diffrents. </returns>
	bool operator!=(const CMatrixClassifierMDMRebias& obj) const { return !isEqual(obj); }

	/// <summary>	Override the ostream operator. </summary>
	/// <param name="os">	The ostream. </param>
	/// <param name="obj">	The object. </param>
	/// <returns>	Return the modified ostream. </returns>
	friend std::ostream& operator <<(std::ostream& os, const CMatrixClassifierMDMRebias& obj)
	{
		os << obj.print().str();		
		return os;
	}


	//*********************
	//***** Variables *****
	//*********************
	/// <summary>	Rebias Matrix. </summary>
	Eigen::MatrixXd m_Rebias;
	/// <summary>	Number of classify launch. </summary>
	size_t m_NbClassify = 0;

protected:

	/// <summary>	Save Additionnal informations (reference and number of classification). </summary>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	bool saveAdditional(tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* data) const override;

	/// <summary>	Load Additionnal informations (reference and number of classification). </summary>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	bool loadAdditional(tinyxml2::XMLElement* data) override;

	/// <summary>	Prints the Additional informations (reference and number of classification). </summary>
	/// <returns>	Additional informations in stringstream</returns>
	std::stringstream printAdditional() const override;
};