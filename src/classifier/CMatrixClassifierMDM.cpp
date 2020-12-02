#include "geometry/classifier/CMatrixClassifierMDM.hpp"
#include "geometry/Mean.hpp"
#include "geometry/Distance.hpp"
#include "geometry/Basics.hpp"
#include "geometry/Geodesic.hpp"
#include <unsupported/Eigen/MatrixFunctions> // SQRT of Matrix

namespace Geometry {

//***********************	
//***** Constructor *****	
//***********************
///-------------------------------------------------------------------------------------------------
CMatrixClassifierMDM::CMatrixClassifierMDM(const size_t nbClass, const EMetric metric)
{
	CMatrixClassifierMDM::setClassCount(nbClass);
	m_metric = metric;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
CMatrixClassifierMDM::~CMatrixClassifierMDM()
{
	m_means.clear();
	m_nbTrials.clear();
}
///-------------------------------------------------------------------------------------------------

//**********************
//***** Classifier *****
//**********************
///-------------------------------------------------------------------------------------------------
void CMatrixClassifierMDM::setClassCount(const size_t nbClass)
{
	if (m_nbClass != nbClass || m_means.size() != nbClass || m_nbTrials.size() != nbClass)
	{
		IMatrixClassifier::setClassCount(nbClass);
		m_means.resize(m_nbClass);
		m_nbTrials.resize(nbClass);
	}
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets)
{
	if (datasets.empty()) { return false; }
	setClassCount(datasets.size());							// Change the number of classes if needed
	for (size_t k = 0; k < m_nbClass; ++k)					// for each class
	{
		if (!Mean(datasets[k], m_means[k], m_metric)) { return false; }	// Compute the mean of each class
		m_nbTrials[k] = datasets[k].size();
	}
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::classify(const Eigen::MatrixXd& sample, size_t& classId, std::vector<double>& distance,
									std::vector<double>& probability, const EAdaptations adaptation, const size_t& realClassId)
{
	if (!IsSquare(sample)) { return false; }				// Verification if it's a square matrix 
	double distMin = std::numeric_limits<double>::max();	// Init of distance min

	// Compute Distances
	distance.resize(m_nbClass);
	for (size_t k = 0; k < m_nbClass; ++k)
	{
		distance[k] = Distance(sample, m_means[k], m_metric);
		if (distMin > distance[k])
		{
			classId = k;
			distMin = distance[k];
		}
	}

	// Compute Probabilities (personnal method)
	probability.resize(m_nbClass);
	double sumProbability = 0.0;
	for (size_t k = 0; k < m_nbClass; ++k)
	{
		probability[k] = distMin / distance[k];
		sumProbability += probability[k];
	}

	for (auto& p : probability) { p /= sumProbability; }

	// Adaptation
	if (adaptation == EAdaptations::None) { return true; }
	// Get class id for adaptation and increase number of trials, expected if supervised, predicted if unsupervised
	const size_t id = adaptation == EAdaptations::Supervised ? realClassId : classId;
	if (id >= m_nbClass) { return false; }					// Check id (if supervised and bad input)
	m_nbTrials[id]++;										// Update number of trials for the class id
	return Geodesic(m_means[id], sample, m_means[id], m_metric, 1.0 / m_nbTrials[id]);
}
///-------------------------------------------------------------------------------------------------

//***********************
//***** XML Manager *****
//***********************
///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::saveClasses(tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* data) const
{
	for (size_t k = 0; k < m_nbClass; ++k)								// for each class
	{
		tinyxml2::XMLElement* element = doc.NewElement("Class");		// Create class node
		element->SetAttribute("class-id", int(k));						// Set attribute class id (0 to K)
		element->SetAttribute("nb-trials", int(m_nbTrials[k]));			// Set attribute class number of trials
		if (!saveMatrix(element, m_means[k])) { return false; }			// Save class Matrix Reference
		data->InsertEndChild(element);									// Add class node to data node
	}
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::loadClasses(tinyxml2::XMLElement* data)
{
	tinyxml2::XMLElement* element = data->FirstChildElement("Class");	// Get Fist Class Node
	for (size_t k = 0; k < m_nbClass; ++k)								// for each class
	{
		if (element == nullptr) { return false; }						// Check if Node Exist
		const size_t idx = element->IntAttribute("class-id");			// Get Id (normally idx == k)
		if (idx != k) { return false; }									// Check Id
		m_nbTrials[k] = element->IntAttribute("nb-trials");				// Get the number of Trials for this class
		if (!loadMatrix(element, m_means[k])) { return false; }			// Load Class Matrix
		element = element->NextSiblingElement("Class");					// Next Class
	}
	return true;
}
///-------------------------------------------------------------------------------------------------

//*****************************
//***** Override Operator *****
//*****************************
///-------------------------------------------------------------------------------------------------
std::stringstream CMatrixClassifierMDM::printClasses() const
{
	std::stringstream ss;
	for (size_t i = 0; i < m_nbClass; ++i)
	{
		ss << "Mean of class " << i << " (" << m_nbTrials[i] << " trials): ";
		if (m_means[i].size() != 0) { ss << std::endl << m_means[i].format(MATRIX_FORMAT) << std::endl; }
		else { ss << "Not Computed" << std::endl; }
	}
	return ss;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::isEqual(const CMatrixClassifierMDM& obj, const double precision) const
{
	if (!IMatrixClassifier::isEqual(obj)) { return false; }
	if (m_nbClass != obj.getClassCount()) { return false; }
	for (size_t i = 0; i < m_nbClass; ++i)
	{
		if (!AreEquals(m_means[i], obj.m_means[i], precision)) { return false; }
		if (m_nbTrials[i] != obj.m_nbTrials[i]) { return false; }
	}
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CMatrixClassifierMDM::copy(const CMatrixClassifierMDM& obj)
{
	IMatrixClassifier::copy(obj);
	setClassCount(m_nbClass);
	for (size_t i = 0; i < m_nbClass; ++i)
	{
		m_means[i]    = obj.m_means[i];
		m_nbTrials[i] = obj.m_nbTrials[i];
	}
}
///-------------------------------------------------------------------------------------------------

}  // namespace Geometry
