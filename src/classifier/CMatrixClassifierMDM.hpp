#pragma once
#include "IMatrixClassifier.hpp"
#include "utils/Metrics.hpp"

class CMatrixClassifierMDM : public IMatrixClassifier
{
public:
	std::vector<Eigen::MatrixXd> m_Means;
	EMetrics m_Metric = Metric_Riemann;

	//***** Constructor *****
	CMatrixClassifierMDM();
	explicit CMatrixClassifierMDM(size_t classcount, EMetrics metric);
	~CMatrixClassifierMDM() override = default;

	//***** Classifier *****
	void setClassCount(const size_t classcount) override;
	bool train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets) override;
	bool classify(const Eigen::MatrixXd& sample, size_t& classid) override;
	bool classify(const Eigen::MatrixXd& sample, size_t& classid, std::vector<double>& distance, std::vector<double>& probability) override;

	//***** XML *****
	bool saveXML(const std::string& filename) override;
	bool loadXML(const std::string& filename) override;

	bool saveHeaderAttribute(tinyxml2::XMLElement* element) const;
	bool loadHeaderAttribute(tinyxml2::XMLElement* element);
	bool saveClass(tinyxml2::XMLElement* element, const size_t index) const;
	bool loadClass(tinyxml2::XMLElement* element, const size_t index);


	//***** Override Operator *****
	bool operator==(const CMatrixClassifierMDM& obj) const;
	bool operator!=(const CMatrixClassifierMDM& obj) const;

	std::string getType() const override { return "Minimum Distance to Mean"; }

	std::stringstream print() const override;
	friend std::ostream& operator <<(std::ostream& os, const CMatrixClassifierMDM& obj);
};
