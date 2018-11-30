#include "CMatrixClassifierFgMDM.hpp"


CMatrixClassifierFgMDM::CMatrixClassifierFgMDM(const size_t classcount)
{
	CMatrixClassifierFgMDM::setClassCount(classcount);
}
///-------------------------------------------------------------------------------------------------

bool CMatrixClassifierFgMDM::train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets)
{
	(void)datasets;
	return true;
}
///-------------------------------------------------------------------------------------------------

bool CMatrixClassifierFgMDM::classify(const Eigen::MatrixXd& sample, uint32_t& classid, std::vector<double>& distance, std::vector<double>& probability)
{
	(void)sample;	(void)classid;	(void)distance;	(void)probability;
	return true;
}
///-------------------------------------------------------------------------------------------------

void CMatrixClassifierFgMDM::setClassCount(const size_t classcount)
{
	IMatrixClassifier::setClassCount(classcount);
}
///-------------------------------------------------------------------------------------------------
