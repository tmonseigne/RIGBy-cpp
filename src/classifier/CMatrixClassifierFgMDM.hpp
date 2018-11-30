#pragma once
#include "IMatrixClassifier.hpp"

class CMatrixClassifierFgMDM : public IMatrixClassifier
{
public:
	CMatrixClassifierFgMDM() = default;
	explicit CMatrixClassifierFgMDM(const size_t classcount);
	~CMatrixClassifierFgMDM() override = default;
	
	void setClassCount(const size_t classcount) override;

	bool train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets) override;
	bool classify(const Eigen::MatrixXd& sample, uint32_t& classid, std::vector<double>& distance, std::vector<double>& probability) override;
	
};

