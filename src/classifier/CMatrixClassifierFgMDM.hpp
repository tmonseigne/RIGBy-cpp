///-------------------------------------------------------------------------------------------------
/// 
/// \file CMatrixClassifierMDM.hpp
/// 
/// \brief Class of Minimum Distance to Mean with geodesic filtering (FgMDM) Classifier
/// 
/// \author Thibaut Monseigne (Inria).
/// 
/// \version 1.0.
/// 
/// \date 10/12/2018.
/// 
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include "IMatrixClassifier.hpp"

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Class of Minimum Distance to Mean with geodesic filtering (FgMDM) Classifier. </summary>
/// 
/// <seealso cref="IMatrixClassifier" />
/// 
///-------------------------------------------------------------------------------------------------
class CMatrixClassifierFgMDM : public IMatrixClassifier
{
public:
	//***********************	
	//***** Constructor *****
	//***********************	
	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Initializes a new instance of the <see cref="CMatrixClassifierFgMDM"/> class. </summary>
	///
	///----------------------------------------------------------------------------------------------------
	CMatrixClassifierFgMDM() = default;
	explicit CMatrixClassifierFgMDM(const size_t classcount);
	~CMatrixClassifierFgMDM() override = default;
	
	//**********************
	//***** Classifier *****
	//**********************
	void setClassCount(const size_t classcount) override;
	bool train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets) override;
	bool classify(const Eigen::MatrixXd& sample, size_t& classid) override;
	bool classify(const Eigen::MatrixXd& sample, size_t& classid, std::vector<double>& distance, std::vector<double>& probability) override;

	//***********************
	//***** XML Manager *****
	//***********************
	bool saveXML(const std::string& filename) override;
	bool loadXML(const std::string& filename) override;

	bool saveHeaderAttribute(tinyxml2::XMLElement* element) const override;
	bool loadHeaderAttribute(tinyxml2::XMLElement* element) override;
	bool saveClass(tinyxml2::XMLElement* element, const size_t index) const override;
	bool loadClass(tinyxml2::XMLElement* element, const size_t index) override;

	//*****************************
	//***** Override Operator *****
	//*****************************
	bool operator==(const CMatrixClassifierFgMDM& obj) const;
	bool operator!=(const CMatrixClassifierFgMDM& obj) const;
	std::string getType() const override { return "Minimum Distance to Mean with geodesic filtering"; }
	std::stringstream print() const override;
	friend std::ostream& operator <<(std::ostream& os, const CMatrixClassifierFgMDM& obj);
};
