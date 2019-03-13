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
CMatrixClassifierMDM::CMatrixClassifierMDM()
{
	CMatrixClassifierMDM::setClassCount(m_classCount);
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
CMatrixClassifierMDM::CMatrixClassifierMDM(const size_t classcount, const EMetrics metric)
{
	CMatrixClassifierMDM::setClassCount(classcount);
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
void CMatrixClassifierMDM::setClassCount(const size_t classcount)
{
	if (m_classCount != classcount || m_Means.size() != classcount || m_NbTrials.size() != classcount)
	{
		IMatrixClassifier::setClassCount(classcount);
		m_Means.resize(m_classCount);
		m_NbTrials.resize(classcount);
	}
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::train(const vector<vector<MatrixXd>>& datasets)
{
	setClassCount(datasets.size());										// Change the number of classes if needed
	for (size_t i = 0; i < m_classCount; ++i)
	{
		if (!Mean(datasets[i], m_Means[i], m_Metric)) { return false; }	// Compute the mean of each class
		m_NbTrials[i] = datasets[i].size();
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

	// Compute Distance
	distance.resize(m_classCount);
	for (size_t i = 0; i < m_classCount; ++i)
	{
		distance[i] = Distance(sample, m_Means[i], m_Metric);
		if (distMin > distance[i])
		{
			classId = i;
			distMin = distance[i];
		}
	}

	// Compute Probability (personnal method)
	probability.resize(m_classCount);
	double sumProbability = 0.0;
	for (size_t i = 0; i < m_classCount; ++i)
	{
		probability[i] = distMin / distance[i];
		sumProbability += probability[i];
	}

	for (auto& p : probability) { p /= sumProbability; }

	// Adaptation
	if (adaptation == Adaptation_None) { return true; }
	// Get class id for adaptation and increase number of trials, expected if supervised, predicted if unsupervised
	const size_t id = adaptation == Adaptation_Supervised ? realClassId : classId;
	if (id >= m_classCount) { return false; }	// Check id (if supervised and bad input)
	m_NbTrials[id]++;							// Update number of trials for the class id
	return Geodesic(m_Means[id], sample, m_Means[id], m_Metric, 1.0 / m_NbTrials[id]);
}
///-------------------------------------------------------------------------------------------------

//***********************
//***** XML Manager *****
//***********************
///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::saveClasses(XMLDocument& doc, XMLElement* data) const
{
	for (size_t k = 0; k < m_classCount; ++k)						// for each class
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
bool CMatrixClassifierMDM::loadClasses(XMLDocument& /*doc*/, XMLElement* data)
{

	XMLElement* element = data->FirstChildElement("Class");			// Get Fist Class Node
	for (size_t k = 0; k < m_classCount; ++k)						// for each class
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

///-------------------------------------------------------------------------------------------------
std::stringstream CMatrixClassifierMDM::printClasses() const
{
	stringstream ss;
	for (size_t i = 0; i < m_classCount; ++i)
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
	if (m_classCount != obj.getClassCount()) { return false; }
	for (size_t i = 0; i < m_classCount; ++i)
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
	setClassCount(m_classCount);
	for (size_t i = 0; i < m_classCount; ++i)
	{
		m_Means[i] = obj.m_Means[i];
		m_NbTrials[i] = obj.m_NbTrials[i];
	}
}
///-------------------------------------------------------------------------------------------------
