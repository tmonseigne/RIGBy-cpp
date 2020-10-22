#include "geometry/classifier/CMatrixClassifierFgMDMRTRebias.hpp"

#include "geometry/Mean.hpp"
#include "geometry/Covariance.hpp"
#include <unsupported/Eigen/MatrixFunctions> // SQRT of Matrix

namespace Geometry {

//**********************
//***** Classifier *****
//**********************
///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDMRTRebias::train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets)
{
	if (!m_bias.computeBias(datasets, m_Metric)) { return false; }
	std::vector<std::vector<Eigen::MatrixXd>> newDatasets;
	m_bias.applyBias(datasets, newDatasets);
	if (!CMatrixClassifierFgMDMRT::train(newDatasets)) { return false; }	// Train FgMDM
	const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(m_ref.rows(), m_ref.cols());	// Identity matrix
	if (AreEquals(m_ref, identity)) { m_ref = identity; }	// Normally it's always the case with Identity matrix we simplify future operation
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDMRTRebias::classify(const Eigen::MatrixXd& sample, size_t& classId, std::vector<double>& distance,
											  std::vector<double>& probability, const EAdaptations adaptation, const size_t& realClassId)
{
	if (!IsSquare(sample)) { return false; }				// Verification if it's a square matrix
	Eigen::MatrixXd newSample;
	m_bias.applyBias(sample, newSample);
	m_bias.updateBias(sample, m_Metric);
	return CMatrixClassifierFgMDMRT::classify(newSample, classId, distance, probability, adaptation, realClassId);
}
///-------------------------------------------------------------------------------------------------

//***********************
//***** XML Manager *****
//***********************
///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDMRTRebias::saveAdditional(tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* data) const
{
	if (!CMatrixClassifierFgMDMRT::saveAdditional(doc, data)) { return false; }
	if (!m_bias.saveAdditional(doc, data)) { return false; }
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDMRTRebias::loadAdditional(tinyxml2::XMLElement* data)
{
	if (!CMatrixClassifierFgMDMRT::loadAdditional(data)) { return false; }
	if (!m_bias.loadAdditional(data)) { return false; }
	return true;
}
///-------------------------------------------------------------------------------------------------

//*****************************
//***** Override Operator *****
//*****************************
///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDMRTRebias::isEqual(const CMatrixClassifierFgMDMRTRebias& obj, const double precision) const
{
	return CMatrixClassifierFgMDMRT::isEqual(obj, precision) && m_bias == obj.m_bias;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CMatrixClassifierFgMDMRTRebias::copy(const CMatrixClassifierFgMDMRTRebias& obj)
{
	CMatrixClassifierFgMDMRT::copy(obj);
	m_bias = obj.m_bias;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
std::stringstream CMatrixClassifierFgMDMRTRebias::printAdditional() const
{
	std::stringstream ss = CMatrixClassifierFgMDMRT::printAdditional();
	ss << m_bias;
	return ss;
}
///-------------------------------------------------------------------------------------------------

}  // namespace Geometry
