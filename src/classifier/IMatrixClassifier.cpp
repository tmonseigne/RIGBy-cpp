#include "geometry/classifier/IMatrixClassifier.hpp"
#include <iostream>

namespace Geometry {

//***********************	
//***** Constructor *****	
//***********************
///-------------------------------------------------------------------------------------------------
IMatrixClassifier::IMatrixClassifier(const size_t nbClass, const EMetric metric)
{
	IMatrixClassifier::setClassCount(nbClass);
	m_Metric = metric;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void IMatrixClassifier::setClassCount(const size_t nbClass) { m_nbClass = nbClass; }
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool IMatrixClassifier::classify(const Eigen::MatrixXd& sample, size_t& classId, const EAdaptations adaptation, const size_t& realClassId)
{
	std::vector<double> distance, probability;
	return classify(sample, classId, distance, probability, adaptation, realClassId);
}
///-------------------------------------------------------------------------------------------------

//***********************
//***** XML Manager *****
//***********************
///-------------------------------------------------------------------------------------------------
bool IMatrixClassifier::saveXML(const std::string& filename) const
{
	tinyxml2::XMLDocument xmlDoc;
	// Create Root
	tinyxml2::XMLNode* root = xmlDoc.NewElement("Classifier");		// Create root node
	xmlDoc.InsertFirstChild(root);									// Add root to XML

	tinyxml2::XMLElement* data = xmlDoc.NewElement("Classifier-data");	// Create data node
	if (!saveHeader(data)) { return false; }						// Save Header attribute
	if (!saveAdditional(xmlDoc, data)) { return false; }			// Save Optionnal Informations
	if (!saveClasses(xmlDoc, data)) { return false; }				// Save Classes

	root->InsertEndChild(data);										// Add data to root
	return xmlDoc.SaveFile(filename.c_str()) == 0;					// save XML (if != 0 it means error)
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool IMatrixClassifier::loadXML(const std::string& filename)
{
	// Load File
	tinyxml2::XMLDocument xmlDoc;
	if (xmlDoc.LoadFile(filename.c_str()) != 0) { return false; }	// Check File Exist and Loading

	// Load Root
	tinyxml2::XMLNode* root = xmlDoc.FirstChild();					// Get Root Node
	if (root == nullptr) { return false; }							// Check Root Node Exist

	// Load Data
	tinyxml2::XMLElement* data = root->FirstChildElement("Classifier-data");	// Get Data Node
	if (!loadHeader(data)) { return false; }						// Load Header attribute
	if (!loadAdditional(data)) { return false; }					// Load Optionnal Informations
	if (!loadClasses(data)) { return false; }						// Load Classes

	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool IMatrixClassifier::convertMatrixToXMLFormat(const Eigen::MatrixXd& in, std::stringstream& out)
{
	out << in.format(MATRIX_FORMAT);
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool IMatrixClassifier::convertXMLFormatToMatrix(std::stringstream& in, Eigen::MatrixXd& out, const size_t rows, const size_t cols)
{
	out = Eigen::MatrixXd::Identity(rows, cols);				// Init With Identity Matrix (in case of)
	for (size_t i = 0; i < rows; ++i)					// Fill Matrix
	{
		for (size_t j = 0; j < cols; ++j) { in >> out(i, j); }
	}
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool IMatrixClassifier::saveMatrix(tinyxml2::XMLElement* element, const Eigen::MatrixXd& matrix)
{
	element->SetAttribute("size", int(matrix.rows()));	// Set Matrix size NxN
	std::stringstream ss;
	convertMatrixToXMLFormat(matrix, ss);
	element->SetText(ss.str().c_str());					// Write Means Value
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool IMatrixClassifier::loadMatrix(tinyxml2::XMLElement* element, Eigen::MatrixXd& matrix)
{
	const size_t size = element->IntAttribute("size");	// Get number of row/col
	if (size == 0) { return true; }
	std::stringstream ss(element->GetText());			// String stream to parse Matrix value
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
	return m_Metric == obj.m_Metric && m_nbClass == obj.getClassCount();
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void IMatrixClassifier::copy(const IMatrixClassifier& obj)
{
	m_Metric = obj.m_Metric;
	setClassCount(obj.getClassCount());
}
/// -------------------------------------------------------------------------------------------------

/// -------------------------------------------------------------------------------------------------
std::stringstream IMatrixClassifier::print() const { return std::stringstream(printHeader().str() + printAdditional().str() + printClasses().str()); }
/// -------------------------------------------------------------------------------------------------

/// -------------------------------------------------------------------------------------------------
std::stringstream IMatrixClassifier::printHeader() const
{
	std::stringstream ss;
	ss << getType() << " Classifier" << std::endl;
	ss << "Metric : " << toString(m_Metric) << std::endl;
	ss << "Number of Classes : " << m_nbClass << std::endl;
	return ss;
}
///-------------------------------------------------------------------------------------------------

//*******************************************
//***** XML Manager (Private Functions) *****
//*******************************************
///-------------------------------------------------------------------------------------------------
bool IMatrixClassifier::saveHeader(tinyxml2::XMLElement* data) const
{
	data->SetAttribute("type", getType().c_str());				// Set attribute classifier type
	data->SetAttribute("class-count", int(m_nbClass));			// Set attribute class count
	data->SetAttribute("metric", toString(m_Metric).c_str());	// Set attribute metric
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool IMatrixClassifier::loadHeader(tinyxml2::XMLElement* data)
{
	if (data == nullptr) { return false; }						// Check if Node Exist
	const std::string classifierType = data->Attribute("type");	// Get type
	if (classifierType != getType()) { return false; }			// Check Type
	setClassCount(data->IntAttribute("class-count"));			// Update Number of classes
	m_Metric = StringToMetric(data->Attribute("metric"));		// Update Metric
	return true;
}
///-------------------------------------------------------------------------------------------------

}  // namespace Geometry
