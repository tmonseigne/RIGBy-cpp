#pragma once
#include "IMatrixClassifier.hpp"

class CMatrixClassifierFgMDM : public IMatrixClassifier
{
public:
	//***** Constructor *****
	CMatrixClassifierFgMDM() = default;
	explicit CMatrixClassifierFgMDM(const size_t classcount);
	~CMatrixClassifierFgMDM() override = default;
	
	//***** Classifier *****
	void setClassCount(const size_t classcount) override;
	bool train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets) override;
	bool classify(const Eigen::MatrixXd& sample, size_t& classid) override;
	bool classify(const Eigen::MatrixXd& sample, size_t& classid, std::vector<double>& distance, std::vector<double>& probability) override;

	//***** XML *****
	bool saveXML(const std::string& filename) override;
	bool loadXML(const std::string& filename) override;

	bool saveHeaderAttribute(tinyxml2::XMLElement* element) const override;
	bool loadHeaderAttribute(tinyxml2::XMLElement* element) override;
	bool saveClass(tinyxml2::XMLElement* element, const size_t index) const override;
	bool loadClass(tinyxml2::XMLElement* element, const size_t index) override;

	//***** Override Operator *****
	bool operator==(const CMatrixClassifierFgMDM& obj) const;
	bool operator!=(const CMatrixClassifierFgMDM& obj) const;
	std::string getType() const override { return "Minimum Distance to Mean with geodesic filtering"; }
	std::stringstream print() const override;
	friend std::ostream& operator <<(std::ostream& os, const CMatrixClassifierFgMDM& obj);
};

