#include "CMatrixClassifierFgMDMRTRebias.hpp"
#include "utils/Mean.hpp"
#include "utils/Covariance.hpp"
#include <unsupported/Eigen/MatrixFunctions> // SQRT of Matrix

using namespace std;
using namespace Eigen;
using namespace tinyxml2;

//**********************
//***** Classifier *****
//**********************
///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDMRTRebias::train(const vector<vector<MatrixXd>>& datasets)
{
	if (!m_Rebias.computeBias(datasets, m_Metric)) { return false; }
	vector<vector<MatrixXd>> newDatasets;
	m_Rebias.applyBias(datasets, newDatasets);
	if (!CMatrixClassifierFgMDMRT::train(newDatasets)) { return false; }		// Train FgMDM
	const MatrixXd identity = MatrixXd::Identity(m_Ref.rows(), m_Ref.cols());	// Identity matrix
	if (AreEquals(m_Ref, identity)) { m_Ref = identity; }						// Normally it's always the case with Identity matrix we simplify future operation
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDMRTRebias::classify(const MatrixXd& sample, size_t& classId, std::vector<double>& distance,
											  std::vector<double>& probability, const EAdaptations adaptation, const size_t& realClassId)
{
	if (!IsSquare(sample)) { return false; }							// Verification if it's a square matrix 
	MatrixXd newSample;
	m_Rebias.applyBias(sample, newSample);
	m_Rebias.updateBias(sample, m_Metric);
	return CMatrixClassifierFgMDMRT::classify(newSample, classId, distance, probability, adaptation, realClassId);
}
///-------------------------------------------------------------------------------------------------

//***********************
//***** XML Manager *****
//***********************
///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDMRTRebias::saveAdditional(XMLDocument& doc, XMLElement* data) const
{
	if (!CMatrixClassifierFgMDMRT::saveAdditional(doc, data)) { return false; }
	if (!m_Rebias.saveAdditional(doc, data)) { return false; }
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDMRTRebias::loadAdditional(XMLElement* data)
{
	if (!CMatrixClassifierFgMDMRT::loadAdditional(data)) { return false; }
	if (!m_Rebias.loadAdditional(data)) { return false; }
	return true;
}
///-------------------------------------------------------------------------------------------------

//*****************************
//***** Override Operator *****
//*****************************
///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDMRTRebias::isEqual(const CMatrixClassifierFgMDMRTRebias& obj, const double precision) const
{
	return CMatrixClassifierFgMDMRT::isEqual(obj, precision) && m_Rebias == obj.m_Rebias;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CMatrixClassifierFgMDMRTRebias::copy(const CMatrixClassifierFgMDMRTRebias& obj)
{
	CMatrixClassifierFgMDMRT::copy(obj);
	m_Rebias = obj.m_Rebias;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
std::stringstream CMatrixClassifierFgMDMRTRebias::printAdditional() const
{
	stringstream ss = CMatrixClassifierFgMDMRT::printAdditional();
	ss << m_Rebias;
	return ss;
}
///-------------------------------------------------------------------------------------------------
