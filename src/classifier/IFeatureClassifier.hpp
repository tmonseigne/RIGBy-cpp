#pragma once

#include <Eigen/Dense>
#include <vector>

class IFeatureClassifier
{
public:

	IFeatureClassifier() = default;
	virtual ~IFeatureClassifier() = default;

	virtual bool train(const std::vector<std::vector<Eigen::RowVectorXd>>& datasets) = 0;
	virtual bool classify(const Eigen::RowVectorXd& sample, uint32_t& classid) = 0;
	virtual bool classify(const Eigen::RowVectorXd& sample, uint32_t& classid, std::vector<double>& distance, std::vector<double>& probability) = 0;

};
