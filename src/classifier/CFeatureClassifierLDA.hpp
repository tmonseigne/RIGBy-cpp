#pragma once

#include "IFeatureClassifier.hpp"
#include <Eigen/Dense>
#include <vector>

class CFeatureClassifierLDA : public IFeatureClassifier
{
public:
	//***** Constructor *****
	CFeatureClassifierLDA() = default;
	~CFeatureClassifierLDA() = default;

	//***** Classifier *****
	bool train(const std::vector<std::vector<Eigen::RowVectorXd>>& datasets) override;
	bool classify(const Eigen::RowVectorXd& sample, size_t& classid) override;
	bool classify(const Eigen::RowVectorXd& sample, size_t& classid, std::vector<double>& distance, std::vector<double>& probability) override;

	//***** XML *****

	//***** Override Operator *****

};
