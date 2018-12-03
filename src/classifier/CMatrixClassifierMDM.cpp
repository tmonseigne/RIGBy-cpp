#include "CMatrixClassifierMDM.hpp"
#include "utils/Mean.hpp"
#include "utils/Distance.hpp"
#include "utils/Basics.hpp"
#include "3rd-party/tinyxml2.h"
#include <iostream>

using namespace std;
using namespace Eigen;
using namespace tinyxml2;

CMatrixClassifierMDM::CMatrixClassifierMDM()
{
	m_Means.resize(m_ClassCount);
}

CMatrixClassifierMDM::CMatrixClassifierMDM(const size_t classcount, const EMetrics metric)
{
	CMatrixClassifierMDM::setClassCount(classcount);
	m_Metric = metric;
}
///-------------------------------------------------------------------------------------------------

void CMatrixClassifierMDM::setClassCount(const size_t classcount)
{
	if (m_ClassCount != classcount)
	{
		IMatrixClassifier::setClassCount(classcount);
		m_Means.resize(m_ClassCount);
	}
}

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

bool CMatrixClassifierMDM::classify(const MatrixXd& sample, uint32_t& classid)
{
	std::vector<double> distance, probability;
	return classify(sample, classid, distance, probability);
}
///-------------------------------------------------------------------------------------------------


bool CMatrixClassifierMDM::classify(const MatrixXd& sample, uint32_t& classid, vector<double>& distance, vector<double>& probability)
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
			classid = uint32_t(i);
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

bool CMatrixClassifierMDM::saveXML(const std::string& filename)
{
	XMLDocument xmlDoc;
	XMLNode* root = xmlDoc.NewElement("Classifier");					// Create root node
	xmlDoc.InsertFirstChild(root);										// Add root to XML

	XMLElement* data = xmlDoc.NewElement("Classifier-data");			// Create data node
	data->SetAttribute("type", "MDM");									// Set attribute classifier type
	data->SetAttribute("class-count", int(m_ClassCount));				// Set attribute class count
	data->SetAttribute("metric", MetricToString(m_Metric).c_str());		// Set attribute metric
	for (size_t i = 0; i < m_ClassCount; ++i)							// for each class
	{
		XMLElement* classElement = xmlDoc.NewElement("Class");			// Create class node
		classElement->SetAttribute("class-id", int(i));					// Set attribute class id (0 to K)
		classElement->SetAttribute("size", int(m_Means[i].rows()));		// Set Matrix size NxN

		IOFormat fmt(FullPrecision, 0, " ", "\n", "", "", "", "");
		stringstream ss;
		ss << m_Means[i].format(fmt);
		classElement->SetText(ss.str().c_str());						// Write Means Value
		data->InsertEndChild(classElement);								// Add class node to data node
	}
	root->InsertEndChild(data);											// Add data to root

	return xmlDoc.SaveFile(filename.c_str()) == 0;						// save XML (if != 0 it means error)
}

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
	if (data == nullptr) { return false; }								// Check Data Node Exist
	const string classifierType = data->Attribute("type");
	if (classifierType != "MDM") { return false; }						// Check Type
	setClassCount(data->IntAttribute("class-count"));					// Update Number of class
	m_Metric = StringToMetric(data->Attribute("metric"));				// Update Metric

	XMLElement* classElement = data->FirstChildElement("Class");		// Get Fist Class Node
	for (size_t k = 0; k < m_ClassCount && classElement != nullptr; ++k)	//Check if Node exist (in case of)
	{
		const size_t idx = classElement->IntAttribute("class-id"),		// Get Id (normally idx = k)
					 size = classElement->IntAttribute("size");			// Get number of row/col
		m_Means[idx] = MatrixXd::Ones(size, size);						// Init With Identity Matrix (in case of)
		std::stringstream iss(classElement->GetText());					// String stream to parse Matrix value
		for (size_t i = 0; i < size; ++i)								// Fill Matrix
		{
			for (size_t j = 0; j < size; ++j)
			{
				iss >> m_Means[idx](i, j);
			}
		}
		classElement = classElement->NextSiblingElement("Class");		// Next Class
	}
	return true;
}
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

bool CMatrixClassifierMDM::operator!=(const CMatrixClassifierMDM& obj) const
{
	return !(*this == obj);
}
///-------------------------------------------------------------------------------------------------
ostream& operator<<(ostream& os, const CMatrixClassifierMDM& obj)
{
	os << "Metric : " << MetricToString(obj.m_Metric) << endl
		<< "Nb of Class : " << obj.m_ClassCount << endl;
	for (size_t i = 0; i < obj.m_ClassCount; ++i)
	{
		os << "Mean Class " << i << " : ";
		if (obj.m_Means[i].size() != 0)
		{
			os << endl << obj.m_Means[i] << endl;
		}
		else
		{
			os << "Not Compute" << endl;
		}
	}
	return os;
}
