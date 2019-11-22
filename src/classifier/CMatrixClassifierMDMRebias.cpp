#include "CMatrixClassifierMDMRebias.hpp"
#include "utils/Mean.hpp"
#include "utils/Basics.hpp"
#include <unsupported/Eigen/MatrixFunctions> // SQRT of Matrix

using namespace std;
using namespace Eigen;
using namespace tinyxml2;


//**********************
//***** Classifier *****
//**********************
///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDMRebias::train(const vector<vector<MatrixXd>>& datasets)
{
	if (!m_Rebias.computeBias(datasets, m_Metric)) { return false; }
	vector<vector<MatrixXd>> newDatasets;
	m_Rebias.applyBias(datasets, newDatasets);
	return CMatrixClassifierMDM::train(newDatasets);					// Train MDM
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDMRebias::classify(const MatrixXd& sample, size_t& classId, std::vector<double>& distance,
										  std::vector<double>& probability, const EAdaptations adaptation, const size_t& realClassId)
{
	if (!IsSquare(sample)) { return false; }							// Verification if it's a square matrix 
	MatrixXd newSample;
	m_Rebias.applyBias(sample, newSample);
	m_Rebias.updateBias(sample, m_Metric);
	return CMatrixClassifierMDM::classify(newSample, classId, distance, probability, adaptation, realClassId);
}
///-------------------------------------------------------------------------------------------------

//***********************
//***** XML Manager *****
//***********************
///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDMRebias::saveAdditional(XMLDocument& doc, XMLElement* data) const
{
	if (!CMatrixClassifierMDM::saveAdditional(doc, data)) { return false; }
	if (!m_Rebias.saveAdditional(doc, data)) { return false; }
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDMRebias::loadAdditional(XMLElement* data)
{
	if (!CMatrixClassifierMDM::loadAdditional(data)) { return false; }
	if (!m_Rebias.loadAdditional(data)) { return false; }
	return true;
}
///-------------------------------------------------------------------------------------------------

//*****************************
//***** Override Operator *****
//*****************************
///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDMRebias::isEqual(const CMatrixClassifierMDMRebias& obj, const double precision) const
{
	return CMatrixClassifierMDM::isEqual(obj, precision) && m_Rebias == obj.m_Rebias;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CMatrixClassifierMDMRebias::copy(const CMatrixClassifierMDMRebias& obj)
{
	CMatrixClassifierMDM::copy(obj);
	m_Rebias = obj.m_Rebias;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
std::stringstream CMatrixClassifierMDMRebias::printAdditional() const
{
	stringstream ss = CMatrixClassifierMDM::printAdditional();
	ss << m_Rebias;
	return ss;
}
///-------------------------------------------------------------------------------------------------
