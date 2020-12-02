///-------------------------------------------------------------------------------------------------
/// 
/// \file CMatrixClassifierFgMDMRT.hpp
/// \brief Class of Minimum Distance to Mean with geodesic filtering (FgMDM) Classifier RT (adaptation is Real Time Assumed)
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 10/12/2018.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
///
///-------------------------------------------------------------------------------------------------

#pragma once

#include "geometry/classifier/CMatrixClassifierMDM.hpp"

namespace Geometry {

/// <summary>	Class of Minimum Distance to Mean with geodesic filtering (FgMDM) Classifier RT (adaptation is Real Time Assumed). </summary>
/// <seealso cref="CMatrixClassifierMDM" />
class CMatrixClassifierFgMDMRT : public CMatrixClassifierMDM
{
public:
	//***********************	
	//***** Constructor *****
	//***********************	
	/// <summary>	Initializes a new instance of the <see cref="CMatrixClassifierFgMDMRT"/> class. </summary>
	CMatrixClassifierFgMDMRT() = default;

	/// <summary>	Default Copy constructor. Initializes a new instance of the <see cref="CMatrixClassifierFgMDMRT"/> class. </summary>
	/// <param name="obj">	Initial object. </param>
	CMatrixClassifierFgMDMRT(const CMatrixClassifierFgMDMRT& obj) { *this = obj; }

	/// <summary>	Initializes a new instance of the <see cref="CMatrixClassifierFgMDMRT"/> class and set base members. </summary>
	/// <param name="nbClass">	The number of classes. </param>
	/// <param name="metric">	Metric to use to calculate means (see also <see cref="EMetric" />). </param>
	explicit CMatrixClassifierFgMDMRT(const size_t nbClass, const EMetric metric) : CMatrixClassifierMDM(nbClass, metric) { }

	/// <summary>	Finalizes an instance of the <see cref="CMatrixClassifierFgMDMRT"/> class. </summary>
	/// <remarks>	clear the <see cref="m_means"/> vector of Matrix. </remarks>
	~CMatrixClassifierFgMDMRT() override = default;

	//***************************
	//***** Getter / Setter *****
	//***************************
	const Eigen::MatrixXd& getRef() const { return m_ref; }					///< Get reference of tangent space. 	
	void setRef(const Eigen::MatrixXd& ref) { m_ref = ref; }				///< Set reference of tangent space. 

	const Eigen::MatrixXd& getWeight() const { return m_weight; }			 ///< Get weight matrix of geodesic filter. 
	void setWeight(const Eigen::MatrixXd& weight) { m_weight = weight; }	 ///< Set weight matrix of geodesic filter. 

	//**********************
	//***** Classifier *****
	//**********************
	/// <summary>	Train the classifier with the dataset.
	/// -# Compute the Riemann mean of all trials as reference and store this in <see cref="m_ref"/> member.
	/// -# Set the good number of classes
	/// -# Trasnform data to the Tangent Space with the reference
	/// -# Compute the FgDA Weight (<see cref="FgDACompute" />).
	/// -# Apply the FgDA Weight and return to Original Manifold.
	/// -# Apply the train function of MDM Classifier (see <see cref="CMatrixClassifierMDM::train"/>)
	///	</summary>
	/// <param name="datasets">	The dataset one class by row and trials on colums. </param>
	/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
	bool train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets) override;

	/// <summary>	Classify the matrix and return the class id, the distance and the probability of each class.\n
	/// -# Transform the sample to the Tangent Space.\n
	/// -# Apply the FgDA weight.\n
	/// -# Return to the original Manifold.\n
	/// -# Apply the classify function of MDM Classifier (see <see cref="CMatrixClassifierMDM::classify"/>)
	///	</summary>
	/// <remarks>
	/// <b>Remark</b> : We use the MDM classification whatever the adaptation method chosen. 
	/// Thus the MDM part evolves but the geodesic filtering does not evolve to keep an execution online. 
	///	A version allowing the adaptation of the Filter will be implemented for offline execution.
	/// </remarks>
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
	bool isEqual(const CMatrixClassifierFgMDMRT& obj, double precision = 1e-6) const;

	/// <summary>	Copy object value. </summary>
	/// <param name="obj">	The object to copy. </param>
	void copy(const CMatrixClassifierFgMDMRT& obj);

	/// <summary>	Get the type of the classifier. </summary>
	/// <returns>	Minimum Distance to Mean with geodesic filtering (FgMDM). </returns>
	std::string getType() const override { return toString(EMatrixClassifiers::FgMDM_RT); }

	/// <summary>	Override the affectation operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	The copied object. </returns>
	CMatrixClassifierFgMDMRT& operator=(const CMatrixClassifierFgMDMRT& obj)
	{
		copy(obj);
		return *this;
	}

	/// <summary>	Override the equal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	<c>True</c> if the two <see cref="CMatrixClassifierFgMDMRT"/> are equals. </returns>
	bool operator==(const CMatrixClassifierFgMDMRT& obj) const { return isEqual(obj); }

	/// <summary>	Override the not equal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	<c>True</c> if the two <see cref="CMatrixClassifierFgMDMRT"/> are diffrents. </returns>
	bool operator!=(const CMatrixClassifierFgMDMRT& obj) const { return !isEqual(obj); }

	/// <summary>	Override the ostream operator. </summary>
	/// <param name="os">	The ostream. </param>
	/// <param name="obj">	The object. </param>
	/// <returns>	Return the modified ostream. </returns>
	friend std::ostream& operator <<(std::ostream& os, const CMatrixClassifierFgMDMRT& obj)
	{
		os << obj.print().str();
		return os;
	}

protected:
	//***********************
	//***** XML Manager *****
	//***********************
	/// <summary>	Save Additionnal informations (Reference and LDA Weight). </summary>
	/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
	bool saveAdditional(tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* data) const override;

	/// <summary>	Load Additionnal informations (Reference and LDA Weight). </summary>
	/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
	bool loadAdditional(tinyxml2::XMLElement* data) override;

	/// <summary>	Prints the Additional informations (Reference and LDA Weight). </summary>
	/// <returns>	Additional informations in stringstream. </returns>
	std::stringstream printAdditional() const override;

	//*********************
	//***** Variables *****
	//*********************
	Eigen::MatrixXd m_ref;		///< Reference matrix of tanget space.
	Eigen::MatrixXd m_weight;	///< Weght matrix of Filter Geodesic Discriminant Analysis.
};

}  // namespace Geometry
