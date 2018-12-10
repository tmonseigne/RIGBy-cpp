#include "CFeatureClassifierLDA.hpp"

bool CFeatureClassifierLDA::train(const std::vector<std::vector<Eigen::RowVectorXd>>& datasets)
{
	return true;
}
bool CFeatureClassifierLDA::classify(const Eigen::RowVectorXd& sample, size_t& classid)
{
	return true;
}
bool CFeatureClassifierLDA::classify(const Eigen::RowVectorXd& sample, size_t& classid, std::vector<double>& distance, std::vector<double>& probability)
{
	return true;
}
