///-------------------------------------------------------------------------------------------------
/// 
/// \file CMatrixClassifierMDMRebias.hpp
/// \brief Class of Minimum Distance to Mean (MDM) Classifier with Rebias.
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 10/12/2018.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include "geometry/classifier/CMatrixClassifierMDM.hpp"
#include "geometry/Metrics.hpp"
#include "geometry/classifier/CBias.hpp"

namespace Geometry {

/// <summary>	Class of Minimum Distance to Mean (MDM) Classifier with Rebias. </summary>
/// <seealso cref="IMatrixClassifier" />
class CMatrixClassifierMDMRebias final : public CMatrixClassifierMDM
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

	/// <summary>	Initializes a new instance of the <see cref="CMatrixClassifierMDMRebias"/> class and set base members. </summary>
	/// \copydetails CMatrixClassifierMDM(const size_t, const EMetric)
	explicit CMatrixClassifierMDMRebias(const size_t nbClass, const EMetric metric) : CMatrixClassifierMDM(nbClass, metric) { }

	/// <summary>	Finalizes an instance of the <see cref="CMatrixClassifierMDMRebias"/> class. </summary>
	~CMatrixClassifierMDMRebias() override = default;

	//***************************
	//***** Getter / Setter *****
	//***************************
	const CBias& getBias() const { return m_bias; }
	void setBias(const CBias& bias) { m_bias = bias; }
	
	//**********************
	//***** Classifier *****
	//**********************

	/// \copybrief IMatrixClassifier::train(const std::vector<std::vector<Eigen::MatrixXd>>&)
	/// <summary>	
	/// -# Compute the mean of all trials with the metric (<see cref="EMetric" />) in <see cref="m_Metric"/> member as reference and store this in <see cref="m_bias"/> member.
	/// -# Set the good number of classes
	/// -# Apply an affine transformation on each trials with the reference : \f$ S_\text{new} = R^{-1/2} * S * {R^{-1/2}}^{\mathsf{T}} \f$
	/// -# Compute the mean of each class (row), on transformed trials, with the metric (<see cref="EMetric" />) in <see cref="m_Metric"/> member.
	/// -# Set the number of trials for each class.
	///	</summary>
	/// \copydetails IMatrixClassifier::train(const std::vector<std::vector<Eigen::MatrixXd>>&)
	bool train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets) override;

	/// \copybrief CMatrixClassifierMDM::classify(const Eigen::MatrixXd&, size_t&, std::vector<double>&, std::vector<double>&, const EAdaptations, const size_t&)
	/// <summary>
	/// Apply an affine transformation on the trial (sample) with the reference : \f$ S_\text{new} = R^{-1/2} * S * {R^{-1/2}}^{\mathsf{T}} \f$ \n
	/// Update the reference with the current sample the first time and next with the Geodesic between the reference and the current sample.\n
	/// With \f$ \gamma_m \f$ the Geodesic (<see cref="Geodesic" />) with the metric \f$ m \f$ (<see cref="EMetric" />) and \f$ N_c \f$ the number of classification : \f$ R = \gamma_\text{m}\left( R,S,\frac{1}{N_c} \right) \f$ \n
	///	</summary>
	/// \copydetails CMatrixClassifierMDM::classify(const Eigen::MatrixXd&, size_t&, std::vector<double>&, std::vector<double>&, const EAdaptations, const size_t&)
	bool classify(const Eigen::MatrixXd& sample, size_t& classId, std::vector<double>& distance, std::vector<double>& probability,
				  EAdaptations adaptation = EAdaptations::None, const size_t& realClassId = std::numeric_limits<size_t>::max()) override;

	//*****************************
	//***** Override Operator *****
	//*****************************

	/// \copydoc CMatrixClassifierMDM::isEqual(const CMatrixClassifierMDM&, const double) const
	bool isEqual(const CMatrixClassifierMDMRebias& obj, double precision = 1e-6) const;

	/// \copydoc CMatrixClassifierMDM::copy(const CMatrixClassifierMDM&)
	void copy(const CMatrixClassifierMDMRebias& obj);

	/// \copybrief IMatrixClassifier::getType()
	/// <returns>	Minimum Distance to Mean REBIAS. </returns>
	std::string getType() const override { return toString(EMatrixClassifiers::MDM_Rebias); }

	/// <summary>	Override the affectation operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	The copied object. </returns>
	CMatrixClassifierMDMRebias& operator=(const CMatrixClassifierMDMRebias& obj)
	{
		copy(obj);
		return *this;
	}

	/// <summary>	Override the egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	<c>True</c> if the two <see cref="CMatrixClassifierMDMRebias"/> are equals. </returns>
	bool operator==(const CMatrixClassifierMDMRebias& obj) const { return isEqual(obj); }

	/// <summary>	Override the not egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	<c>True</c> if the two <see cref="CMatrixClassifierMDMRebias"/> are diffrents. </returns>
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

protected:

	/// <summary>	Save Additionnal informations (reference and number of classification). </summary>
	/// <returns>	<c>True</c> if it succeeds, <c>false</c> otherwise. </returns>
	bool saveAdditional(tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* data) const override;

	/// <summary>	Load Additionnal informations (reference and number of classification). </summary>
	/// <returns>	<c>True</c> if it succeeds, <c>false</c> otherwise. </returns>
	bool loadAdditional(tinyxml2::XMLElement* data) override;

	/// <summary>	Prints the Additional informations (reference and number of classification). </summary>
	/// <returns>	Additional informations in stringstream</returns>
	std::stringstream printAdditional() const override;

	//*********************
	//***** Variables *****
	//*********************
	/// <summary>	Rebias Method. </summary>
	CBias m_bias;
};

}  // namespace Geometry
