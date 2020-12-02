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
	/// <param name="nbClass">	The number of classes. </param>
	/// <param name="metric">	Metric to use to calculate means (see also <see cref="EMetric" />). </param>
	explicit CMatrixClassifierMDMRebias(const size_t nbClass, const EMetric metric) : CMatrixClassifierMDM(nbClass, metric) { }

	/// <summary>	Finalizes an instance of the <see cref="CMatrixClassifierMDMRebias"/> class. </summary>
	~CMatrixClassifierMDMRebias() override = default;

	//***************************
	//***** Getter / Setter *****
	//***************************
	const CBias& getBias() const { return m_bias; }		///< Get Rebias Method. 
	void setBias(const CBias& bias) { m_bias = bias; }	///< Set Rebias Method. 
	
	//**********************
	//***** Classifier *****
	//**********************

	/// <summary>	Train the classifier with the dataset.
	/// -# Compute the mean of all trials with the metric (<see cref="EMetric" />) in <see cref="m_metric"/> member as reference and store this in <see cref="m_bias"/> member.
	/// -# Set the good number of classes
	/// -# Apply an affine transformation on each trials with the reference : \f$ S_\text{new} = R^{-1/2} * S * {R^{-1/2}}^{\mathsf{T}} \f$
	/// -# Compute the mean of each class (row), on transformed trials, with the metric (<see cref="EMetric" />) in <see cref="m_metric"/> member.
	/// -# Set the number of trials for each class.
	///	</summary>
	/// <param name="datasets">	The dataset one class by row and trials on colums. </param>
	/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
	bool train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets) override;

	/// <summary>	Classify the matrix and return the class id, the distance and the probability of each class.
	/// -# Apply an affine transformation on the trial (sample) with the reference : \f$ S_\text{new} = R^{-1/2} * S * {R^{-1/2}}^{\mathsf{T}} \f$
	/// -# Update the reference with the current sample the first time and next with the Geodesic between the reference and the current sample.\n
	/// With \f$ \gamma_m \f$ the Geodesic (<see cref="Geodesic" />) with the metric \f$ m \f$ (<see cref="EMetric" />) and \f$ N_c \f$ the number of classification : \f$ R = \gamma_\text{m}\left( R,S,\frac{1}{N_c} \right) \f$
	/// -# Apply the classify function of MDM Classifier (see <see cref="CMatrixClassifierMDM::classify"/>)
	///	</summary>
	/// \copydetails IMatrixClassifier::classify(const Eigen::MatrixXd&, size_t&, std::vector<double>&, std::vector<double>&, const EAdaptations, const size_t&)
	bool classify(const Eigen::MatrixXd& sample, size_t& classId, std::vector<double>& distance, std::vector<double>& probability,
				  EAdaptations adaptation = EAdaptations::None, const size_t& realClassId = std::numeric_limits<size_t>::max()) override;

	//*****************************
	//***** Override Operator *****
	//*****************************

	/// <summary>	Check if object are equals (with a precision tolerance). </summary>
	/// <param name="obj">			The second object. </param>
	/// <param name="precision">	Precision for matrix comparison. </param>
	/// <returns>	<c>True</c> if the two elements are equals (with a precision tolerance). </returns>
	bool isEqual(const CMatrixClassifierMDMRebias& obj, double precision = 1e-6) const;

	/// <summary>	Copy object value. </summary>
	/// <param name="obj">	The object to copy. </param>
	void copy(const CMatrixClassifierMDMRebias& obj);

	/// <summary>	Get the type of the classifier. </summary>
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

	/// <summary>	Override the equal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	<c>True</c> if the two <see cref="CMatrixClassifierMDMRebias"/> are equals. </returns>
	bool operator==(const CMatrixClassifierMDMRebias& obj) const { return isEqual(obj); }

	/// <summary>	Override the not equal operator. </summary>
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
	/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
	bool saveAdditional(tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* data) const override;

	/// <summary>	Load Additionnal informations (reference and number of classification). </summary>
	/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
	bool loadAdditional(tinyxml2::XMLElement* data) override;

	/// <summary>	Prints the Additional informations (reference and number of classification). </summary>
	/// <returns>	Additional informations in stringstream. </returns>
	std::stringstream printAdditional() const override;

	//*********************
	//***** Variables *****
	//*********************
	CBias m_bias;	///< Rebias Method. 
};

}  // namespace Geometry
