#include "CMatrixClassifierMDM.hpp"
#include "utils/Mean.hpp"
#include "utils/Distance.hpp"
#include "utils/Basics.hpp"
#include "utils/Geodesic.hpp"
#include <unsupported/Eigen/MatrixFunctions> // SQRT of Matrix

using namespace std;
using namespace Eigen;
using namespace tinyxml2;

//***********************	
//***** Constructor *****	
//***********************
///-------------------------------------------------------------------------------------------------
CMatrixClassifierMDM::CMatrixClassifierMDM() { CMatrixClassifierMDM::setClassCount(m_nbClass); }
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
CMatrixClassifierMDM::CMatrixClassifierMDM(const size_t nbClass, const EMetrics metric)
{
	CMatrixClassifierMDM::setClassCount(nbClass);
	m_Metric = metric;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
CMatrixClassifierMDM::~CMatrixClassifierMDM()
{
	m_Means.clear();
	m_NbTrials.clear();
}
///-------------------------------------------------------------------------------------------------

//**********************
//***** Classifier *****
//**********************
///-------------------------------------------------------------------------------------------------
void CMatrixClassifierMDM::setClassCount(const size_t nbClass)
{
	if (m_nbClass != nbClass || m_Means.size() != nbClass || m_NbTrials.size() != nbClass)
	{
		IMatrixClassifier::setClassCount(nbClass);
		m_Means.resize(m_nbClass);
		m_NbTrials.resize(nbClass);
	}
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::train(const vector<vector<MatrixXd>>& datasets)
{
	if (datasets.empty()) { return false; }
	setClassCount(datasets.size());										// Change the number of classes if needed
	for (size_t k = 0; k < m_nbClass; ++k)								// for each class
	{
		if (!Mean(datasets[k], m_Means[k], m_Metric)) { return false; }	// Compute the mean of each class
		m_NbTrials[k] = datasets[k].size();
	}
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::classify(const MatrixXd& sample, size_t& classId, std::vector<double>& distance,
									std::vector<double>& probability, const EAdaptations adaptation, const size_t& realClassId)
{
	if (!isSquare(sample)) { return false; }				// Verification if it's a square matrix 
	double distMin = std::numeric_limits<double>::max();	// Init of distance min

	// Compute Distances
	distance.resize(m_nbClass);
	for (size_t k = 0; k < m_nbClass; ++k)
	{
		distance[k] = Distance(sample, m_Means[k], m_Metric);
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
	if (adaptation == Adaptation_None) { return true; }
	// Get class id for adaptation and increase number of trials, expected if supervised, predicted if unsupervised
	const size_t id = adaptation == Adaptation_Supervised ? realClassId : classId;
	if (id >= m_nbClass) { return false; }					// Check id (if supervised and bad input)
	m_NbTrials[id]++;										// Update number of trials for the class id
	return Geodesic(m_Means[id], sample, m_Means[id], m_Metric, 1.0 / m_NbTrials[id]);
}
///-------------------------------------------------------------------------------------------------

//***********************
//***** XML Manager *****
//***********************
///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::saveClasses(XMLDocument& doc, XMLElement* data) const
{
	for (size_t k = 0; k < m_nbClass; ++k)							// for each class
	{
		XMLElement* element = doc.NewElement("Class");				// Create class node
		element->SetAttribute("class-id", int(k));					// Set attribute class id (0 to K)
		element->SetAttribute("nb-trials", int(m_NbTrials[k]));		// Set attribute class number of trials
		if (!saveMatrix(element, m_Means[k])) { return false; }		// Save class Matrix Reference
		data->InsertEndChild(element);								// Add class node to data node
	}
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::loadClasses(XMLElement* data)
{
	XMLElement* element = data->FirstChildElement("Class");			// Get Fist Class Node
	for (size_t k = 0; k < m_nbClass; ++k)							// for each class
	{
		if (element == nullptr) { return false; }					// Check if Node Exist
		const size_t idx = element->IntAttribute("class-id");		// Get Id (normally idx == k)
		if (idx != k) { return false; }								// Check Id
		m_NbTrials[k] = element->IntAttribute("nb-trials");			// Get the number of Trials for this class
		if (!loadMatrix(element, m_Means[k])) { return false; }		// Load Class Matrix
		element = element->NextSiblingElement("Class");				// Next Class
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
	stringstream ss;
	for (size_t i = 0; i < m_nbClass; ++i)
	{
		ss << "Mean of class " << i << " (" << m_NbTrials[i] << " trials): ";
		if (m_Means[i].size() != 0) { ss << endl << m_Means[i] << endl; }
		else { ss << "Not Computed" << endl; }
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
		if (!AreEquals(m_Means[i], obj.m_Means[i], precision)) { return false; }
		if (m_NbTrials[i] != obj.m_NbTrials[i]) { return false; }
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
		m_Means[i]    = obj.m_Means[i];
		m_NbTrials[i] = obj.m_NbTrials[i];
	}
}
///-------------------------------------------------------------------------------------------------
