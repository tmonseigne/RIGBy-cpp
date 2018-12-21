#include "CMatrixClassifierMDM.hpp"
#include "utils/Mean.hpp"
#include "utils/Distance.hpp"
#include "utils/Basics.hpp"
#include <iostream>

using namespace std;
using namespace Eigen;
using namespace tinyxml2;

///-------------------------------------------------------------------------------------------------
CMatrixClassifierMDM::CMatrixClassifierMDM()
{
	m_Means.resize(m_ClassCount);
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
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CMatrixClassifierMDM::setClassCount(const size_t classcount)
{
	if (m_ClassCount != classcount)
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
	if (!isSquare(sample)) { return false; }
	double distMin = std::numeric_limits<double>::max();
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

	for (auto& p : probability)
	{
		p /= sumProbability;
	}
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::saveXML(const std::string& filename)
{
	XMLDocument xmlDoc;
	XMLNode* root = xmlDoc.NewElement("Classifier");					// Create root node
	xmlDoc.InsertFirstChild(root);										// Add root to XML

	XMLElement* data = xmlDoc.NewElement("Classifier-data");			// Create data node
	if (!saveHeaderAttribute(data)) { return false; }
	for (size_t k = 0; k < m_ClassCount; ++k)							// for each class
	{
		XMLElement* classElement = xmlDoc.NewElement("Class");			// Create class node
		if (!saveClass(classElement, k)) { return false; }
		data->InsertEndChild(classElement);								// Add class node to data node
	}
	root->InsertEndChild(data);											// Add data to root

	return xmlDoc.SaveFile(filename.c_str()) == 0;						// save XML (if != 0 it means error)
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::loadXML(const std::string& filename)
{
	// Load File
	XMLDocument xmlDoc;
	if (xmlDoc.LoadFile(filename.c_str()) != 0) { return false; }		// Check File Exist and Loading

	// Load Root
	XMLNode* root = xmlDoc.FirstChild();								// Get Root Node
	if (root == nullptr) { return false; }								// Check Root Node Exist

	// Load Data
	XMLElement* data = root->FirstChildElement("Classifier-data");		// Get Data Node
	if (!loadHeaderAttribute(data)) { return false; }

	XMLElement* classElement = data->FirstChildElement("Class");		// Get Fist Class Node
	for (size_t k = 0; k < m_ClassCount; ++k)							// for each class
	{
		if (!loadClass(classElement, k)) { return false; }
		classElement = classElement->NextSiblingElement("Class");		// Next Class
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
	if (element == nullptr) { return false; }							// Check if Node Exist
	const string classifierType = element->Attribute("type");			// Get type
	if (classifierType != "MDM") { return false; }						// Check Type
	setClassCount(element->IntAttribute("class-count"));				// Update Number of class
	m_Metric = StringToMetric(element->Attribute("metric"));			// Update Metric
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::saveClass(XMLElement* element, const size_t index) const
{
	element->SetAttribute("class-id", int(index));					// Set attribute class id (0 to K)
	element->SetAttribute("size", int(m_Means[index].rows()));		// Set Matrix size NxN

	const IOFormat fmt(FullPrecision, 0, " ", "\n", "", "", "", "");
	stringstream ss;
	ss << m_Means[index].format(fmt);
	element->SetText(ss.str().c_str());								// Write Means Value
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::loadClass(XMLElement* element, const size_t index)
{
	if (element == nullptr) { return false; }					// Check if Node Exist
	const size_t idx = element->IntAttribute("class-id"),		// Get Id (normally idx = k)
				 size = element->IntAttribute("size");			// Get number of row/col
	if (idx != index) { return false; }							// Check if the file is well parsed
	m_Means[idx] = MatrixXd::Identity(size, size);				// Init With Identity Matrix (in case of)
	std::stringstream iss(element->GetText());					// String stream to parse Matrix value
	for (size_t i = 0; i < size; ++i)							// Fill Matrix
	{
		for (size_t j = 0; j < size; ++j)
		{
			iss >> m_Means[idx](i, j);
		}
	}

	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::operator==(const CMatrixClassifierMDM& obj) const
{
	if (m_Metric != obj.m_Metric || m_ClassCount != obj.m_ClassCount) { return false; }
	for (size_t i = 0; i < m_ClassCount; ++i)
	{
		if (!m_Means[i].isApprox(obj.m_Means[i], 1e-6)) { return false; }
	}
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierMDM::operator!=(const CMatrixClassifierMDM& obj) const
{
	return !(*this == obj);
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
stringstream CMatrixClassifierMDM::print() const
{
	stringstream ss;
	ss << "Metric : " << MetricToString(m_Metric) << endl
		<< "Nb of Class : " << m_ClassCount << endl;
	for (size_t i = 0; i < m_ClassCount; ++i)
	{
		ss << "Mean Class " << i << " : ";
		if (m_Means[i].size() != 0) { ss << endl << m_Means[i] << endl; }
		else { ss << "Not Compute" << endl; }
	}
	return ss;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& os, const CMatrixClassifierMDM& obj)
{
	os << obj.print().str();
	return os;
}
///-------------------------------------------------------------------------------------------------
