#include "geometry/classifier/CMatrixClassifierMDMRebias.hpp"
#include "geometry/Mean.hpp"
#include "geometry/Basics.hpp"
#include <unsupported/Eigen/MatrixFunctions> // SQRT of Matrix

namespace Geometry {

//**********************
//***** Classifier *****
//**********************
///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDMRebias::train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets)
{
	if (!m_bias.computeBias(datasets, m_Metric)) { return false; }
	std::vector<std::vector<Eigen::MatrixXd>> newDatasets;
	m_bias.applyBias(datasets, newDatasets);
	return CMatrixClassifierMDM::train(newDatasets);			// Train MDM
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDMRebias::classify(const Eigen::MatrixXd& sample, size_t& classId, std::vector<double>& distance,
										  std::vector<double>& probability, const EAdaptations adaptation, const size_t& realClassId)
{
	if (!IsSquare(sample)) { return false; }					// Verification if it's a square matrix 
	Eigen::MatrixXd newSample;
	m_bias.applyBias(sample, newSample);
	m_bias.updateBias(sample, m_Metric);
	return CMatrixClassifierMDM::classify(newSample, classId, distance, probability, adaptation, realClassId);
}
///-------------------------------------------------------------------------------------------------

//***********************
//***** XML Manager *****
//***********************
///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDMRebias::saveAdditional(tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* data) const
{
	if (!CMatrixClassifierMDM::saveAdditional(doc, data)) { return false; }
	if (!m_bias.saveAdditional(doc, data)) { return false; }
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDMRebias::loadAdditional(tinyxml2::XMLElement* data)
{
	if (!CMatrixClassifierMDM::loadAdditional(data)) { return false; }
	if (!m_bias.loadAdditional(data)) { return false; }
	return true;
}
///-------------------------------------------------------------------------------------------------

//*****************************
//***** Override Operator *****
//*****************************
///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDMRebias::isEqual(const CMatrixClassifierMDMRebias& obj, const double precision) const
{
	return CMatrixClassifierMDM::isEqual(obj, precision) && m_bias == obj.m_bias;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CMatrixClassifierMDMRebias::copy(const CMatrixClassifierMDMRebias& obj)
{
	CMatrixClassifierMDM::copy(obj);
	m_bias = obj.m_bias;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
std::stringstream CMatrixClassifierMDMRebias::printAdditional() const
{
	std::stringstream ss = CMatrixClassifierMDM::printAdditional();
	ss << m_bias;
	return ss;
}
///-------------------------------------------------------------------------------------------------

}  // namespace Geometry
