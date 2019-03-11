#include "CMatrixClassifierMDMRebias.hpp"
#include "utils/Mean.hpp"
#include "utils/Distance.hpp"
#include "utils/Basics.hpp"
#include "utils/Geodesic.hpp"
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
	//Rebias Reference
	Mean(Vector2DTo1D(datasets), m_Rebias, m_Metric);				// Compute Rebias reference
	const MatrixXd isR = m_Rebias.sqrt().inverse();					// Square root & Inverse Square root of Rebias matrix => isR

	setClassCount(datasets.size());									// Change the number of classes if needed
	for (size_t i = 0; i < m_classCount; ++i)
	{
		m_NbTrials[i] = datasets[i].size();
		vector<MatrixXd> newDatas;									// Create new dataset trasnforme with Rebias
		newDatas.reserve(m_NbTrials[i]);
		for (const auto& data : datasets[i]) { newDatas.emplace_back(isR * data * isR); }	// Transforme dataset for class i
		if (!Mean(newDatas, m_Means[i], m_Metric)) { return false; }	// Compute the mean of each class
	}
	m_NbClassify = 0;												// Used for Rebias Reference adaptation
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDMRebias::classify(const MatrixXd& sample, size_t& classId, std::vector<double>& distance,
										  std::vector<double>& probability, const EAdaptations adaptation, const size_t& realClassId)
{
	if (!isSquare(sample)) { return false; }				// Verification if it's a square matrix 
	double distMin = std::numeric_limits<double>::max();	// Init of distance min
	m_NbClassify++;											// Update number of classify

	// Change sample if Rebias Adaptation
	const MatrixXd isR = m_Rebias.sqrt().inverse(),		// Square root & Inverse Square root of Rebias matrix => isR
				   newSample = isR * sample * isR;		// isC * sample * isC;
	// Modify rebias for the next step
	if (m_NbClassify == 1) { m_Rebias = sample; }		// At the first pass
	else { Geodesic(m_Rebias, sample, m_Rebias, m_Metric, 1.0 / m_NbClassify); }

	// Compute Distance
	distance.resize(m_classCount);
	for (size_t i = 0; i < m_classCount; ++i)
	{
		distance[i] = Distance(newSample, m_Means[i], m_Metric);
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
	return Geodesic(m_Means[id], newSample, m_Means[id], m_Metric, 1.0 / m_NbTrials[id]);
}
///-------------------------------------------------------------------------------------------------

//***** XML Manager *****
//***********************
///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDMRebias::saveXML(const std::string& filename)
{
	XMLDocument xmlDoc;
	// Create Root
	XMLNode* root = xmlDoc.NewElement("Classifier");				// Create root node
	xmlDoc.InsertFirstChild(root);									// Add root to XML

	// Write Header
	XMLElement* data = xmlDoc.NewElement("Classifier-data");		// Create data node
	if (!saveHeaderAttribute(data)) { return false; }				// Save Header attribute

	// Write Class
	for (size_t k = 0; k < m_classCount; ++k)						// for each class
	{
		XMLElement* element = xmlDoc.NewElement("Class");			// Create class node
		element->SetAttribute("class-id", int(k));					// Set attribute class id (0 to K)
		element->SetAttribute("nb-trials", int(m_NbTrials[k]));		// Set attribute class number of trials
		if (!saveMatrix(element, m_Means[k])) { return false; }		// Save class Matrix Reference
		data->InsertEndChild(element);								// Add class node to data node
	}
	root->InsertEndChild(data);										// Add data to root

	return xmlDoc.SaveFile(filename.c_str()) == 0;					// save XML (if != 0 it means error)
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDMRebias::loadXML(const std::string& filename)
{
	// Load File
	XMLDocument xmlDoc;
	if (xmlDoc.LoadFile(filename.c_str()) != 0) { return false; }	// Check File Exist and Loading

	// Load Root
	XMLNode* root = xmlDoc.FirstChild();							// Get Root Node
	if (root == nullptr) { return false; }							// Check Root Node Exist

	// Load Data
	XMLElement* data = root->FirstChildElement("Classifier-data");	// Get Data Node
	if (!loadHeaderAttribute(data)) { return false; }				// Load Header attribute

	XMLElement* element = data->FirstChildElement("Class");			// Get Fist Class Node
	for (size_t k = 0; k < m_classCount; ++k)						// for each class
	{
		if (element == nullptr) { return false; }					// Check if Node Exist
		const size_t idx = element->IntAttribute("class-id");		// Get Id (normally idx = k)
		if (idx != k) { return false; }								// Check Id
		m_NbTrials[k] = element->IntAttribute("nb-trials");			// Get the number of Trials for this class
		if (!loadMatrix(element, m_Means[k])) { return false; }		// Load Class Matrix
		element = element->NextSiblingElement("Class");				// Next Class
	}
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDMRebias::saveHeaderAttribute(XMLElement* element) const
{
	element->SetAttribute("type", "MDM");								// Set attribute classifier type
	element->SetAttribute("class-count", int(m_classCount));			// Set attribute class count
	element->SetAttribute("metric", MetricToString(m_Metric).c_str());	// Set attribute metric
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDMRebias::loadHeaderAttribute(XMLElement* element)
{
	if (element == nullptr) { return false; }						// Check if Node Exist
	const string classifierType = element->Attribute("type");		// Get type
	if (classifierType != "MDM") { return false; }					// Check Type
	setClassCount(element->IntAttribute("class-count"));			// Update Number of classes
	m_Metric = StringToMetric(element->Attribute("metric"));		// Update Metric
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDMRebias::isEqual(const CMatrixClassifierMDMRebias& obj, const double precision) const
{
	return CMatrixClassifierMDM::isEqual(obj) && AreEquals(m_Rebias, obj.m_Rebias, precision) && m_NbClassify == obj.m_NbClassify;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CMatrixClassifierMDMRebias::copy(const CMatrixClassifierMDMRebias& obj)
{
	CMatrixClassifierMDM::copy(obj);
	m_Rebias = obj.m_Rebias;
	m_NbClassify = obj.m_NbClassify;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
stringstream CMatrixClassifierMDMRebias::print() const
{
	stringstream ss = CMatrixClassifierMDM::print();
	ss << "REBIAS Matrix : ";
	if (m_Rebias.size() != 0) { ss << endl << m_Rebias << endl; }
	else { ss << "Not Computed" << endl; }
	ss << "Number of Classification : " << m_NbClassify << endl;
	return ss;
}
///-------------------------------------------------------------------------------------------------
