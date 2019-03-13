///-------------------------------------------------------------------------------------------------
/// 
/// \file CMatrixClassifierFgMDM.hpp
/// \brief Class of Minimum Distance to Mean with geodesic filtering (FgMDM) Classifier
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 10/12/2018.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
///
///-------------------------------------------------------------------------------------------------

#pragma once

#include "CMatrixClassifierMDM.hpp"

/// <summary>	Class of Minimum Distance to Mean with geodesic filtering (FgMDM) Classifier. </summary>
/// <seealso cref="CMatrixClassifierMDM" />
class CMatrixClassifierFgMDM : public CMatrixClassifierMDM
{
public:
	//***********************	
	//***** Constructor *****
	//***********************	
	/// <summary>	Initializes a new instance of the <see cref="CMatrixClassifierFgMDM"/> class. </summary>
	CMatrixClassifierFgMDM() = default;

	/// <summary>	Initializes a new instance of the <see cref="CMatrixClassifierFgMDM"/> class and set base members. </summary>
	/// \copydetails IMatrixClassifier(size_t, EMetrics)
	explicit CMatrixClassifierFgMDM(const size_t classcount, const EMetrics metric) : CMatrixClassifierMDM(classcount, metric) { }

	/// <summary>	Finalizes an instance of the <see cref="CMatrixClassifierFgMDM"/> class. </summary>
	/// <remarks>	clear the <see cref="m_Means"/> vector of Matrix. </remarks>
	~CMatrixClassifierFgMDM() override = default;

	//**********************
	//***** Classifier *****
	//**********************
	/// \copydoc IMatrixClassifier::train(const std::vector<std::vector<Eigen::MatrixXd>>&)
	bool train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets) override;

	/// \copydoc CMatrixClassifierMDM::classify(const Eigen::MatrixXd&, size_t&, std::vector<double>&, std::vector<double>&, bool)
	bool classify(const Eigen::MatrixXd& sample, size_t& classId, std::vector<double>& distance, std::vector<double>& probability,
				  const EAdaptations adaptation = Adaptation_None, const size_t& realClassId = std::numeric_limits<std::size_t>::max()) override;


	//*****************************
	//***** Override Operator *****
	//*****************************
	/// \copybrief IMatrixClassifier::getType()
	/// <returns>	Minimum Distance to Mean with geodesic filtering. </returns>
	std::string getType() const override { return "Minimum Distance to Mean with geodesic filtering"; }

	//***** Variables *****
	Eigen::MatrixXd m_Ref, m_Weight;

protected:
	//*****************************
	//***** Override Operator *****
	//*****************************
	/// \copydoc IMatrixClassifier::isEqual(const IMatrixClassifier&, const double) const
	bool isEqual(const CMatrixClassifierFgMDM& obj, double precision = 1e-6) const;

	/// \copydoc IMatrixClassifier::copy(const IMatrixClassifier&)
	void copy(const CMatrixClassifierFgMDM& obj);

	//***********************
	//***** XML Manager *****
	//***********************
	bool saveAdditional(tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* data) const override;
	bool loadAdditional(tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* data) override;

	std::stringstream printAdditional() const override;
};
