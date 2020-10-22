#include "geometry/classifier/CMatrixClassifierFgMDM.hpp"
#include "geometry/Mean.hpp"

namespace Geometry {

///-------------------------------------------------------------------------------------------------
CMatrixClassifierFgMDM::~CMatrixClassifierFgMDM()
{
	for (auto& v : m_datasets) { v.clear(); }
	m_datasets.clear();
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDM::train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets)
{
	m_datasets = datasets;
	return train();
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDM::classify(const Eigen::MatrixXd& sample, size_t& classId, std::vector<double>& distance,
									  std::vector<double>& probability, const EAdaptations adaptation, const size_t& realClassId)
{
	if (!CMatrixClassifierFgMDMRT::classify(sample, classId, distance, probability, EAdaptations::None)) { return false; }

	// Adaptation
	if (adaptation == EAdaptations::None) { return true; }
	// Get class id for adaptation and increase number of trials, expected if supervised, predicted if unsupervised
	const size_t id = adaptation == EAdaptations::Supervised ? realClassId : classId;
	if (id >= m_nbClass) { return false; }					// Check id (if supervised and bad input)
	m_nbTrials[id]++;										// Update number of trials for the class id
	m_datasets[id].push_back(sample);						// Update the dataset

	// Retrain 
	return train();
}
///-------------------------------------------------------------------------------------------------

}  // namespace Geometry
