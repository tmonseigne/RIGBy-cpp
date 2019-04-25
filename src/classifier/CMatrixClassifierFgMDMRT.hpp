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

#include "CMatrixClassifierMDM.hpp"

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

	/// \copydoc CMatrixClassifierMDM(CMatrixClassifierMDM&&)
	CMatrixClassifierFgMDMRT(CMatrixClassifierFgMDMRT&& obj) = default;

	/// <summary>	Initializes a new instance of the <see cref="CMatrixClassifierFgMDMRT"/> class and set base members. </summary>
	/// \copydetails CMatrixClassifierMDM(size_t, EMetrics)
	explicit CMatrixClassifierFgMDMRT(const size_t nbClass, const EMetrics metric) : CMatrixClassifierMDM(nbClass, metric) { }

	/// <summary>	Finalizes an instance of the <see cref="CMatrixClassifierFgMDMRT"/> class. </summary>
	/// <remarks>	clear the <see cref="m_Means"/> vector of Matrix. </remarks>
	~CMatrixClassifierFgMDMRT() override = default;

	//**********************
	//***** Classifier *****
	//**********************
	/// \copybrief IMatrixClassifier::train(const std::vector<std::vector<Eigen::MatrixXd>>&)
	/// <summary>	
	/// -# Compute the Riemann mean of all trials as reference and store this in <see cref="m_Ref"/> member.
	/// -# Set the good number of classes
	/// -# Trasnform data to the Tangent Space with the reference
	/// -# Compute the FgDA Weight (<see cref="FgDACompute" />).
	/// -# Apply the FgDA Weight and return to Original Manifold.
	/// -# Apply the MDM train (<see cref=" CMatrixClassifierMDM::train(const std::vector<std::vector<Eigen::MatrixXd>>&)" />).
	///	</summary>
	/// \copydetails IMatrixClassifier::train(const std::vector<std::vector<Eigen::MatrixXd>>&)
	bool train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets) override;

	/// \copybrief CMatrixClassifierMDM::classify(const Eigen::MatrixXd&, size_t&, std::vector<double>&, std::vector<double>&, const EAdaptations, const size_t&)
	/// <summary>
	/// -# Transform the sample to the Tangent Space.\n
	/// -# Apply the FgDA weight.\n
	/// -# Return to the original Manifold.\n
	/// -# Apply the MDM classify (<see cref="CMatrixClassifierMDM::classify(const Eigen::MatrixXd&, size_t&, std::vector<double>&, std::vector<double>&, const EAdaptations, const size_t&)"/>).
	///	</summary>
	/// <remarks>
	/// <b>Remark</b> : We use the MDM classification whatever the adaptation method chosen. 
	/// Thus the MDM part evolves but the geodesic filtering does not evolve to keep an execution online. 
	///	A version allowing the adaptation of the Filter will be implemented for offline execution.
	/// </remarks>
	/// \copydetails IMatrixClassifier::classify(const Eigen::MatrixXd&, size_t&, std::vector<double>&, std::vector<double>&, const EAdaptations, const size_t&)
	bool classify(const Eigen::MatrixXd& sample, size_t& classId, std::vector<double>& distance, std::vector<double>& probability,
				  const EAdaptations adaptation = Adaptation_None, const size_t& realClassId = std::numeric_limits<std::size_t>::max()) override;


	//*****************************
	//***** Override Operator *****
	//*****************************
	/// \copydoc CMatrixClassifierMDM::isEqual(const CMatrixClassifierMDM&, const double) const
	bool isEqual(const CMatrixClassifierFgMDMRT& obj, double precision = 1e-6) const;

	/// \copydoc CMatrixClassifierMDM::copy(const CMatrixClassifierMDM&)
	void copy(const CMatrixClassifierFgMDMRT& obj);
	
	/// \copybrief CMatrixClassifierMDM::getType()
	/// <returns>	Minimum Distance to Mean with geodesic filtering. </returns>
	std::string getType() const override { return "Minimum Distance to Mean with geodesic filtering Real Time assumed"; }

	/// <summary>	Override the affectation operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	The copied object. </returns>
	CMatrixClassifierFgMDMRT& operator=(const CMatrixClassifierFgMDMRT& obj)
	{
		copy(obj);
		return *this;
	}

	/// <summary>	Don't Override the move operator. </summary>
	CMatrixClassifierFgMDMRT& operator=(CMatrixClassifierFgMDMRT&& obj) = default;

	/// <summary>	Override the egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	True if the two <see cref="CMatrixClassifierFgMDMRT"/> are equals. </returns>
	bool operator==(const CMatrixClassifierFgMDMRT& obj) const { return isEqual(obj); }

	/// <summary>	Override the not egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	True if the two <see cref="CMatrixClassifierFgMDMRT"/> are diffrents. </returns>
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

	//*********************
	//***** Variables *****
	//*********************
	Eigen::MatrixXd m_Ref, m_Weight;

protected:
	//***********************
	//***** XML Manager *****
	//***********************
	/// <summary>	Save Additionnal informations (Reference and LDA Weight). </summary>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	bool saveAdditional(tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* data) const override;

	/// <summary>	Load Additionnal informations (Reference and LDA Weight). </summary>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	bool loadAdditional(tinyxml2::XMLElement* data) override;

	/// <summary>	Prints the Additional informations (Reference and LDA Weight). </summary>
	/// <returns>	Additional informations in stringstream</returns>
	std::stringstream printAdditional() const override;
};
