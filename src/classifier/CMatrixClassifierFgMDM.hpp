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

#include "CMatrixClassifierFgMDMRT.hpp"

/// <summary>	Class of Minimum Distance to Mean with geodesic filtering (FgMDM) Classifier. </summary>
/// <seealso cref="CMatrixClassifierMDM" />
class CMatrixClassifierFgMDM : public CMatrixClassifierFgMDMRT
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
	/// \copydetails CMatrixClassifierFgMDMRT(size_t, EMetrics)
	explicit CMatrixClassifierFgMDM(const size_t nbClass, const EMetrics metric) : CMatrixClassifierFgMDMRT(nbClass, metric) { }

	/// <summary>	Finalizes an instance of the <see cref="CMatrixClassifierFgMDM"/> class. </summary>
	/// <remarks>	clear the <see cref="m_Means"/> vector of Matrix and the <see cref="m_Datasets"/> member. </remarks>
	~CMatrixClassifierFgMDM() override;

	//**********************
	//***** Classifier *****
	//**********************
	/// \copydoc CMatrixClassifierFgMDMRT::train(const std::vector<std::vector<Eigen::MatrixXd>>&)
	/// <remarks>	the datasets is saved. </remarks>
	bool train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets) override;

	/// \copydoc CMatrixClassifierFgMDMRT::classify(const Eigen::MatrixXd&, size_t&, std::vector<double>&, std::vector<double>&, const EAdaptations, const size_t&)
	/// <remarks>	The classifier is train with the new sample. </remarks>
	bool classify(const Eigen::MatrixXd& sample, size_t& classId, std::vector<double>& distance, std::vector<double>& probability,
				  EAdaptations adaptation = Adaptation_None, const size_t& realClassId = std::numeric_limits<std::size_t>::max()) override;


	//*****************************
	//***** Override Operator *****
	//*****************************

	/// \copybrief CMatrixClassifierMDM::getType()
	/// <returns>	Minimum Distance to Mean with geodesic filtering. </returns>
	std::string getType() const override { return IMatrixClassifier::getType(Matrix_Classifier_FgMDM); }

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

	//*********************
	//***** Variables *****
	//*********************
	///<summary> Data set for train and adaptation (it can quickly rise). </summary>
	std::vector<std::vector<Eigen::MatrixXd>> m_Datasets;

private:
	///<summary> train with the actual datasets (<see cref="m_Datasets"/>). </summary>
	bool train() { return CMatrixClassifierFgMDMRT::train(m_Datasets); }
};
