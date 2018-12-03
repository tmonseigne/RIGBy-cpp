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
	bool classify(const Eigen::MatrixXd& sample, uint32_t& classid) override;
	bool classify(const Eigen::MatrixXd& sample, uint32_t& classid, std::vector<double>& distance, std::vector<double>& probability) override;

	//***** XML *****
	bool saveXML(const std::string& filename) override;
	bool loadXML(const std::string& filename) override;

	//***** Override Operator *****

};

