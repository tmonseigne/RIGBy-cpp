#include "IMatrixClassifier.hpp"
#include <iostream>

using namespace std;
using namespace Eigen;
using namespace tinyxml2;

//***********************	
//***** Constructor *****	
//***********************
///-------------------------------------------------------------------------------------------------
IMatrixClassifier::IMatrixClassifier(const size_t classcount, const EMetrics metric)
{
	IMatrixClassifier::setClassCount(classcount);
	m_Metric = metric;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
IMatrixClassifier::IMatrixClassifier(const IMatrixClassifier& obj)
{
	copy(obj);
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void IMatrixClassifier::setClassCount(const size_t classcount)
{
	m_classCount = classcount;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool IMatrixClassifier::classify(const MatrixXd& sample, size_t& classId, const EAdaptations adaptation, const size_t& realClassId)
{
	vector<double> distance, probability;
	return classify(sample, classId, distance, probability, adaptation, realClassId);
}

bool IMatrixClassifier::saveXML(const string& filename)
{
	XMLDocument xmlDoc;
	// Create Root
	XMLNode* root = xmlDoc.NewElement("Classifier");				// Create root node
	xmlDoc.InsertFirstChild(root);									// Add root to XML

	XMLElement* data = xmlDoc.NewElement("Classifier-data");		// Create data node
	if (!saveHeader(xmlDoc, data)) { return false; }				// Save Header attribute
	if (!saveAdditional(xmlDoc, data)) { return false; }				// Save Optionnal Informations
	if (!saveClasses(xmlDoc, data)) { return false; }				// Save Classes

	root->InsertEndChild(data);										// Add data to root
	return xmlDoc.SaveFile(filename.c_str()) == 0;					// save XML (if != 0 it means error)
}

bool IMatrixClassifier::loadXML(const string& filename)
{
	// Load File
	XMLDocument xmlDoc;
	if (xmlDoc.LoadFile(filename.c_str()) != 0) { return false; }	// Check File Exist and Loading

	// Load Root
	XMLNode* root = xmlDoc.FirstChild();							// Get Root Node
	if (root == nullptr) { return false; }							// Check Root Node Exist

	// Load Data
	XMLElement* data = root->FirstChildElement("Classifier-data");	// Get Data Node
	if (!loadHeader(xmlDoc, data)) { return false; }				// Load Header attribute
	if (!loadAdditional(xmlDoc, data)) { return false; }				// Load Optionnal Informations
	if (!loadClasses(xmlDoc, data)) { return false; }				// Load Classes

	return true;
}
///-------------------------------------------------------------------------------------------------

//***********************
//***** XML Manager *****
//***********************
///-------------------------------------------------------------------------------------------------
bool IMatrixClassifier::convertMatrixToXMLFormat(const MatrixXd& in, stringstream& out)
{
	const IOFormat fmt(FullPrecision, 0, " ", "\n", "", "", "", "");
	out << in.format(fmt);
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool IMatrixClassifier::convertXMLFormatToMatrix(stringstream& in, MatrixXd& out, const size_t rows, const size_t cols)
{
	out = MatrixXd::Identity(rows, cols);				// Init With Identity Matrix (in case of)
	for (size_t i = 0; i < rows; ++i)					// Fill Matrix
	{
		for (size_t j = 0; j < cols; ++j) { in >> out(i, j); }
	}
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool IMatrixClassifier::saveMatrix(XMLElement* element, const MatrixXd& matrix)
{
	element->SetAttribute("size", int(matrix.rows()));	// Set Matrix size NxN
	stringstream ss;
	convertMatrixToXMLFormat(matrix, ss);
	element->SetText(ss.str().c_str());					// Write Means Value
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool IMatrixClassifier::loadMatrix(XMLElement* element, MatrixXd& matrix)
{
	const size_t size = element->IntAttribute("size");	// Get number of row/col
	stringstream ss(element->GetText());				// String stream to parse Matrix value
	convertXMLFormatToMatrix(ss, matrix, size, size);
	return true;
}
///-------------------------------------------------------------------------------------------------

//*****************************
//***** Override Operator *****
//*****************************
///-------------------------------------------------------------------------------------------------
bool IMatrixClassifier::isEqual(const IMatrixClassifier& obj, const double /*precision*/) const
{
	return m_Metric == obj.m_Metric && m_classCount == obj.getClassCount();
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void IMatrixClassifier::copy(const IMatrixClassifier& obj)
{
	m_Metric = obj.m_Metric;
	setClassCount(obj.getClassCount());
}

std::stringstream IMatrixClassifier::print() const
{
	return stringstream(printHeader().str() + printAdditional().str() + printClasses().str());
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
std::stringstream IMatrixClassifier::printHeader() const
{
	stringstream ss;
	ss << getType() << " Classifier" << endl;
	ss << "Metric : " << MetricToString(m_Metric) << endl;
	ss << "Number of Classes : " << m_classCount << endl;
	return ss;
}
///-------------------------------------------------------------------------------------------------

//***********************
//***** XML Manager *****
//***********************
///-------------------------------------------------------------------------------------------------
bool IMatrixClassifier::saveHeader(XMLDocument& /*doc*/, XMLElement* data) const
{
	data->SetAttribute("type", getType().c_str());					// Set attribute classifier type
	data->SetAttribute("class-count", int(m_classCount));			// Set attribute class count
	data->SetAttribute("metric", MetricToString(m_Metric).c_str());	// Set attribute metric
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool IMatrixClassifier::loadHeader(XMLDocument& /*doc*/, XMLElement* data)
{
	if (data == nullptr) { return false; }						// Check if Node Exist
	const string classifierType = data->Attribute("type");		// Get type
	if (classifierType != getType()) { return false; }				// Check Type
	setClassCount(data->IntAttribute("class-count"));			// Update Number of classes
	m_Metric = StringToMetric(data->Attribute("metric"));		// Update Metric
	return true;
}
///-------------------------------------------------------------------------------------------------
