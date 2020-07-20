#include "CMatrixClassifierFgMDM.hpp"
#include "utils/Mean.hpp"

using namespace std;
using namespace Eigen;
using namespace tinyxml2;

///-------------------------------------------------------------------------------------------------
CMatrixClassifierFgMDM::~CMatrixClassifierFgMDM()
{
	for (auto& v : m_Datasets) { v.clear(); }
	m_Datasets.clear();
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDM::train(const std::vector<std::vector<MatrixXd>>& datasets)
{
	m_Datasets = datasets;
	return train();
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDM::classify(const MatrixXd& sample, size_t& classId, std::vector<double>& distance,
									  std::vector<double>& probability, const EAdaptations adaptation, const size_t& realClassId)
{
	if (!CMatrixClassifierFgMDMRT::classify(sample, classId, distance, probability, EAdaptations::None)) { return false; }

	// Adaptation
	if (adaptation == EAdaptations::None) { return true; }
	// Get class id for adaptation and increase number of trials, expected if supervised, predicted if unsupervised
	const size_t id = adaptation == EAdaptations::Supervised ? realClassId : classId;
	if (id >= m_nbClass) { return false; }					// Check id (if supervised and bad input)
	m_NbTrials[id]++;										// Update number of trials for the class id
	m_Datasets[id].push_back(sample);						// Update the dataset

	// Retrain 
	return train();
}
///-------------------------------------------------------------------------------------------------
