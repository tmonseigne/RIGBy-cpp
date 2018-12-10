#include "CFeatureClassifierLDA.hpp"

bool CFeatureClassifierLDA::train(const std::vector<std::vector<Eigen::RowVectorXd>>& datasets)
{
	(void)datasets;
	return true;
}
bool CFeatureClassifierLDA::classify(const Eigen::RowVectorXd& sample, size_t& classid)
{
	std::vector<double> distance, probability;
	return classify(sample, classid, distance, probability);
}
bool CFeatureClassifierLDA::classify(const Eigen::RowVectorXd& sample, size_t& classid, std::vector<double>& distance, std::vector<double>& probability)
{
	(void)sample;
	(void)classid;
	(void)distance;
	(void)probability;
	return true;
}
