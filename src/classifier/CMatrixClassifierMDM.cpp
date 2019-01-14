#include "CMatrixClassifierMDM.hpp"
#include "utils/Mean.hpp"
#include "utils/Distance.hpp"
#include "utils/Basics.hpp"

using namespace std;
using namespace Eigen;
using namespace tinyxml2;

//***********************	
//***** Constructor *****	
//***********************
///-------------------------------------------------------------------------------------------------
CMatrixClassifierMDM::CMatrixClassifierMDM()
{
	CMatrixClassifierMDM::setClassCount(m_ClassCount);
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
CMatrixClassifierMDM::CMatrixClassifierMDM(const CMatrixClassifierMDM& obj)
{
	copy(obj);
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
CMatrixClassifierMDM::~CMatrixClassifierMDM()
{
	m_Means.clear();
}
///-------------------------------------------------------------------------------------------------

//**********************
//***** Classifier *****
//**********************
///-------------------------------------------------------------------------------------------------
void CMatrixClassifierMDM::setClassCount(const size_t classcount)
{
	if (m_ClassCount != classcount || m_Means.size() != classcount)
	{
		IMatrixClassifier::setClassCount(classcount);
		m_Means.resize(m_ClassCount);
	}
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::train(const vector<vector<MatrixXd>>& datasets)
{
	setClassCount(datasets.size());										// Change the number of class if needed
	for (size_t i = 0; i < m_ClassCount; ++i)
	{
		if (!Mean(datasets[i], m_Means[i], m_Metric)) { return false; }	// Compute the mean of each class
	}
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::classify(const MatrixXd& sample, size_t& classid)
{
	std::vector<double> distance, probability;
	return classify(sample, classid, distance, probability);
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::classify(const MatrixXd& sample, size_t& classid, vector<double>& distance, vector<double>& probability)
{
	if (!isSquare(sample)) { return false; }				// Verification if it's a square matrix 
	double distMin = std::numeric_limits<double>::max();	// Init of distance min

	// Compute Distance
	distance.resize(m_ClassCount);
	for (size_t i = 0; i < m_ClassCount; ++i)
	{
		distance[i] = Distance(sample, m_Means[i], m_Metric);
		if (distMin > distance[i])
		{
			classid = i;
			distMin = distance[i];
		}
	}

	// Compute Probability (personnal method)
	probability.resize(m_ClassCount);
	double sumProbability = 0.0;
	for (size_t i = 0; i < m_ClassCount; ++i)
	{
		probability[i] = distMin / distance[i];
		sumProbability += probability[i];
	}

	for (auto& p : probability) { p /= sumProbability; }
	return true;
}
///-------------------------------------------------------------------------------------------------

//***********************
//***** XML Manager *****
//***********************
///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::saveXML(const std::string& filename)
{
	XMLDocument xmlDoc;
	// Create Root
	XMLNode* root = xmlDoc.NewElement("Classifier");				// Create root node
	xmlDoc.InsertFirstChild(root);									// Add root to XML

	// Write Header
	XMLElement* data = xmlDoc.NewElement("Classifier-data");		// Create data node
	if (!saveHeaderAttribute(data)) { return false; }				// Save Header attribute

	// Write Class
	for (size_t k = 0; k < m_ClassCount; ++k)						// for each class
	{
		XMLElement* element = xmlDoc.NewElement("Class");			// Create class node
		element->SetAttribute("class-id", int(k));					// Set attribute class id (0 to K)
		if (!saveMatrix(element, m_Means[k])) { return false; }		// Save class
		data->InsertEndChild(element);								// Add class node to data node
	}
	root->InsertEndChild(data);										// Add data to root

	return xmlDoc.SaveFile(filename.c_str()) == 0;					// save XML (if != 0 it means error)
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::loadXML(const std::string& filename)
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
	for (size_t k = 0; k < m_ClassCount; ++k)						// for each class
	{
		if (element == nullptr) { return false; }					// Check if Node Exist
		const size_t idx = element->IntAttribute("class-id");		// Get Id (normally idx = k)
		if (idx != k) { return false; }								// Check Id
		if (!loadMatrix(element, m_Means[k])) { return false; }		// Load Class Matrix
		element = element->NextSiblingElement("Class");				// Next Class
	}
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::saveHeaderAttribute(XMLElement* element) const
{
	element->SetAttribute("type", "MDM");								// Set attribute classifier type
	element->SetAttribute("class-count", int(m_ClassCount));			// Set attribute class count
	element->SetAttribute("metric", MetricToString(m_Metric).c_str());	// Set attribute metric
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::loadHeaderAttribute(XMLElement* element)
{
	if (element == nullptr) { return false; }						// Check if Node Exist
	const string classifierType = element->Attribute("type");		// Get type
	if (classifierType != "MDM") { return false; }					// Check Type
	setClassCount(element->IntAttribute("class-count"));			// Update Number of class
	m_Metric = StringToMetric(element->Attribute("metric"));		// Update Metric
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::isEqual(const CMatrixClassifierMDM& obj, const double precision) const
{
	if (!IMatrixClassifier::isEqual(obj)) { return false; }
	for (size_t i = 0; i < m_ClassCount; ++i)
	{
		if (!AreEquals(m_Means[i], obj.m_Means[i], precision)) { return false; }
	}
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CMatrixClassifierMDM::copy(const CMatrixClassifierMDM& obj)
{
	IMatrixClassifier::copy(obj);
	setClassCount(m_ClassCount);
	for (size_t i = 0; i < m_ClassCount; ++i) { m_Means[i] = obj.m_Means[i]; }
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
stringstream CMatrixClassifierMDM::print() const
{
	stringstream ss;
	ss << "Metric : " << MetricToString(m_Metric) << endl;
	ss << "Nb of Class : " << m_ClassCount << endl;
	for (size_t i = 0; i < m_ClassCount; ++i)
	{
		ss << "Mean Class " << i << " : ";
		if (m_Means[i].size() != 0) { ss << endl << m_Means[i] << endl; }
		else { ss << "Not Compute" << endl; }
	}
	return ss;
}
///-------------------------------------------------------------------------------------------------
