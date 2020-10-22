///-------------------------------------------------------------------------------------------------
/// 
/// \file CMatrixClassifierFgMDM.hpp
/// \brief Class of Minimum Distance to Mean with geodesic filtering (FgMDM) Classifier.
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 10/12/2018.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
///
///-------------------------------------------------------------------------------------------------

#pragma once

#include "geometry/classifier/CMatrixClassifierFgMDMRT.hpp"

namespace Geometry {

/// <summary>	Class of Minimum Distance to Mean with geodesic filtering (FgMDM) Classifier. </summary>
/// <seealso cref="CMatrixClassifierMDM" />
class CMatrixClassifierFgMDM final : public CMatrixClassifierFgMDMRT
{
public:
	//***********************	
	//***** Constructor *****
	//***********************	
	/// <summary>	Initializes a new instance of the <see cref="CMatrixClassifierFgMDM"/> class. </summary>
	CMatrixClassifierFgMDM() = default;

	/// <summary>	Default Copy constructor. Initializes a new instance of the <see cref="CMatrixClassifierFgMDM"/> class. </summary>
	/// <param name="obj">	Initial object. </param>
	CMatrixClassifierFgMDM(const CMatrixClassifierFgMDM& obj) { *this = obj; }

	/// <summary>	Copy constructor with parent class. Initializes a new instance of the <see cref="CMatrixClassifierFgMDM"/> class. </summary>
	/// <param name="obj">	Initial object. </param>
	explicit CMatrixClassifierFgMDM(const CMatrixClassifierFgMDMRT& obj) { copy(obj); }

	/// <summary>	Initializes a new instance of the <see cref="CMatrixClassifierFgMDM"/> class and set base members. </summary>
	/// \copydetails CMatrixClassifierFgMDMRT(size_t, EMetric)
	explicit CMatrixClassifierFgMDM(const size_t nbClass, const EMetric metric) : CMatrixClassifierFgMDMRT(nbClass, metric) { }

	/// <summary>	Finalizes an instance of the <see cref="CMatrixClassifierFgMDM"/> class. </summary>
	/// <remarks>	clear the <see cref="m_means"/> vector of Matrix and the <see cref="m_datasets"/> member. </remarks>
	~CMatrixClassifierFgMDM() override;

	//***************************
	//***** Getter / Setter *****
	//***************************
	const std::vector<std::vector<Eigen::MatrixXd>>& getDatasets() const { return m_datasets; }
	void setDatasets(const std::vector<std::vector<Eigen::MatrixXd>>& datasets) { m_datasets = datasets; }
	
	//**********************
	//***** Classifier *****
	//**********************
	/// \copydoc CMatrixClassifierFgMDMRT::train(const std::vector<std::vector<Eigen::MatrixXd>>&)
	/// <remarks>	the datasets is saved. </remarks>
	bool train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets) override;

	/// \copydoc CMatrixClassifierFgMDMRT::classify(const Eigen::MatrixXd&, size_t&, std::vector<double>&, std::vector<double>&, const EAdaptations, const size_t&)
	/// <remarks>	The classifier is train with the new sample. </remarks>
	bool classify(const Eigen::MatrixXd& sample, size_t& classId, std::vector<double>& distance, std::vector<double>& probability,
				  EAdaptations adaptation = EAdaptations::None, const size_t& realClassId = std::numeric_limits<size_t>::max()) override;


	//*****************************
	//***** Override Operator *****
	//*****************************

	/// \copybrief CMatrixClassifierMDM::getType()
	/// <returns>	Minimum Distance to Mean with geodesic filtering (FgMDM). </returns>
	std::string getType() const override { return toString(EMatrixClassifiers::FgMDM); }

	/// <summary>	Override the affectation operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	The copied object. </returns>
	CMatrixClassifierFgMDM& operator=(const CMatrixClassifierFgMDM& obj)
	{
		copy(obj);
		return *this;
	}

	/// <summary>	Override the ostream operator. </summary>
	/// <param name="os">	The ostream. </param>
	/// <param name="obj">	The object. </param>
	/// <returns>	Return the modified ostream. </returns>
	friend std::ostream& operator <<(std::ostream& os, const CMatrixClassifierFgMDM& obj)
	{
		os << obj.print().str();
		return os;
	}

protected:
	///<summary> train with the actual datasets (<see cref="m_datasets"/>). </summary>
	bool train() { return CMatrixClassifierFgMDMRT::train(m_datasets); }

	//*********************
	//***** Variables *****
	//*********************
	std::vector<std::vector<Eigen::MatrixXd>> m_datasets;	///< Data set for train and adaptation (it can quickly rise).
};

}  // namespace Geometry
