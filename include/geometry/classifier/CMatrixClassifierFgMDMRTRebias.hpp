///-------------------------------------------------------------------------------------------------
/// 
/// \file CMatrixClassifierFgMDMRTRebias.hpp
/// \brief Class of Minimum Distance to Mean with geodesic filtering (FgMDM) Classifier RT (adaptation is Real Time Assumed)
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 10/12/2018.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
///
///-------------------------------------------------------------------------------------------------

#pragma once

#include "geometry/classifier/CMatrixClassifierFgMDMRT.hpp"
#include "geometry/classifier/CBias.hpp"

namespace Geometry {

/// <summary>	Class of Minimum Distance to Mean with geodesic filtering (FgMDM) Classifier RT (adaptation is Real Time Assumed). </summary>
/// <seealso cref="CMatrixClassifierFgMDMRT" />
class CMatrixClassifierFgMDMRTRebias final : public CMatrixClassifierFgMDMRT
{
public:
	//***********************	
	//***** Constructor *****
	//***********************	
	/// <summary>	Initializes a new instance of the <see cref="CMatrixClassifierFgMDMRTRebias"/> class. </summary>
	CMatrixClassifierFgMDMRTRebias() = default;

	/// <summary>	Default Copy constructor. Initializes a new instance of the <see cref="CMatrixClassifierFgMDMRTRebias"/> class. </summary>
	/// <param name="obj">	Initial object. </param>
	CMatrixClassifierFgMDMRTRebias(const CMatrixClassifierFgMDMRTRebias& obj) { *this = obj; }

	/// <summary>	Initializes a new instance of the <see cref="CMatrixClassifierFgMDMRTRebias"/> class and set base members. </summary>
	/// \copydetails CMatrixClassifierFgMDMRT(size_t, EMetric)
	explicit CMatrixClassifierFgMDMRTRebias(const size_t nbClass, const EMetric metric) : CMatrixClassifierFgMDMRT(nbClass, metric) { }

	/// <summary>	Finalizes an instance of the <see cref="CMatrixClassifierFgMDMRTRebias"/> class. </summary>
	/// <remarks>	clear the <see cref="m_means"/> vector of Matrix. </remarks>
	~CMatrixClassifierFgMDMRTRebias() override = default;

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
	/// -# Compute the Riemann mean of all trials as reference and store this in <see cref="m_ref"/> member.
	/// -# Set the good number of classes
	/// -# Trasnform data to the Tangent Space with the reference
	/// -# Compute the FgDA Weight (<see cref="FgDACompute" />).
	/// -# Apply the FgDA Weight and return to Original Manifold.
	/// -# Apply the MDM train (<see cref=" CMatrixClassifierFgMDMRT::train(const std::vector<std::vector<Eigen::MatrixXd>>&)" />).
	///	</summary>
	/// \copydetails IMatrixClassifier::train(const std::vector<std::vector<Eigen::MatrixXd>>&)
	bool train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets) override;

	/// \copybrief CMatrixClassifierFgMDMRT::classify(const Eigen::MatrixXd&, size_t&, std::vector<double>&, std::vector<double>&, const EAdaptations, const size_t&)
	/// <summary>
	/// -# Transform the sample to the Tangent Space.\n
	/// -# Apply the FgDA weight.\n
	/// -# Return to the original Manifold.\n
	/// -# Apply the MDM classify (<see cref="CMatrixClassifierFgMDMRT::classify(const Eigen::MatrixXd&, size_t&, std::vector<double>&, std::vector<double>&, const EAdaptations, const size_t&)"/>).
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
	/// \copydoc CMatrixClassifierFgMDMRT::isEqual(const CMatrixClassifierFgMDMRT&, const double) const
	bool isEqual(const CMatrixClassifierFgMDMRTRebias& obj, double precision = 1e-6) const;

	/// \copydoc CMatrixClassifierFgMDMRT::copy(const CMatrixClassifierFgMDMRT&)
	void copy(const CMatrixClassifierFgMDMRTRebias& obj);

	/// \copybrief CMatrixClassifierFgMDMRT::getType()
	/// <returns>	Minimum Distance to Mean with geodesic filtering (FgMDM). </returns>
	std::string getType() const override { return toString(EMatrixClassifiers::FgMDM_RT_Rebias); }

	/// <summary>	Override the affectation operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	The copied object. </returns>
	CMatrixClassifierFgMDMRTRebias& operator=(const CMatrixClassifierFgMDMRTRebias& obj)
	{
		copy(obj);
		return *this;
	}

	/// <summary>	Override the egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	<c>True</c> if the two <see cref="CMatrixClassifierFgMDMRTRebias"/> are equals. </returns>
	bool operator==(const CMatrixClassifierFgMDMRTRebias& obj) const { return isEqual(obj); }

	/// <summary>	Override the not egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	<c>True</c> if the two <see cref="CMatrixClassifierFgMDMRTRebias"/> are diffrents. </returns>
	bool operator!=(const CMatrixClassifierFgMDMRTRebias& obj) const { return !isEqual(obj); }

	/// <summary>	Override the ostream operator. </summary>
	/// <param name="os">	The ostream. </param>
	/// <param name="obj">	The object. </param>
	/// <returns>	Return the modified ostream. </returns>
	friend std::ostream& operator <<(std::ostream& os, const CMatrixClassifierFgMDMRTRebias& obj)
	{
		os << obj.print().str();
		return os;
	}

protected:
	//***********************
	//***** XML Manager *****
	//***********************
	/// <summary>	Save Additionnal informations (Reference and LDA Weight). </summary>
	/// <returns>	<c>True</c> if it succeeds, <c>false</c> otherwise. </returns>
	bool saveAdditional(tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* data) const override;

	/// <summary>	Load Additionnal informations (Reference and LDA Weight). </summary>
	/// <returns>	<c>True</c> if it succeeds, <c>false</c> otherwise. </returns>
	bool loadAdditional(tinyxml2::XMLElement* data) override;

	/// <summary>	Prints the Additional informations (Reference and LDA Weight). </summary>
	/// <returns>	Additional informations in stringstream</returns>
	std::stringstream printAdditional() const override;

	//*********************
	//***** Variables *****
	//*********************
	CBias m_bias;	///< Rebias Method. 

};

}  // namespace Geometry
