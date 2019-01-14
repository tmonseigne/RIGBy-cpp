#include "CMatrixClassifierFgMDM.hpp"
#include "utils/Mean.hpp"
#include "utils/Basics.hpp"
#include "utils/Featurization.hpp"
#include "utils/Classification.hpp"

using namespace std;
using namespace Eigen;
using namespace tinyxml2;

///-------------------------------------------------------------------------------------------------
CMatrixClassifierFgMDM::CMatrixClassifierFgMDM(const CMatrixClassifierFgMDM& obj)
{
	copy(obj);
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDM::train(const vector<vector<MatrixXd>>& datasets)
{
	if (datasets.empty()) { return false; }
	// Compute Reference matrix
	const vector<MatrixXd> data = Vector2DTo1D(datasets);		// Append datasets in one vector
	if (!Mean(data, m_Ref, Metric_Riemann)) { return false; }

	// Transform to the Tangent Space
	const size_t nbClass = datasets.size();
	vector<vector<RowVectorXd>> tsSample(nbClass);
	for (size_t k = 0; k < nbClass; ++k)
	{
		const size_t nbTrials = datasets[k].size();
		tsSample[k].resize(nbTrials);
		for (size_t i = 0; i < nbTrials; ++i)
		{
			if (!TangentSpace(datasets[k][i], tsSample[k][i], m_Ref)) { return false; }
		}
	}

	// Compute FgDA Weight
	if (!FgDACompute(tsSample, m_Weight)) { return false; }

	// Convert Datasets
	vector<vector<MatrixXd>> newDatasets(nbClass);
	vector<vector<RowVectorXd>> filtered(nbClass);
	for (size_t k = 0; k < nbClass; ++k)
	{
		const size_t nbTrials = datasets[k].size();
		newDatasets[k].resize(nbTrials);
		filtered[k].resize(nbTrials);
		for (size_t i = 0; i < nbTrials; ++i)
		{
			if (!FgDAApply(tsSample[k][i], filtered[k][i], m_Weight)) { return false; }	// Apply Filter
			if (!UnTangentSpace(filtered[k][i], newDatasets[k][i], m_Ref)) { return false; }	// Return to Matrix Space
		}
	}

	return CMatrixClassifierMDM::train(newDatasets);					// Train MDM
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDM::classify(const MatrixXd& sample, size_t& classid)
{
	std::vector<double> distance, probability;
	return classify(sample, classid, distance, probability);
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDM::classify(const MatrixXd& sample, size_t& classid, vector<double>& distance, vector<double>& probability)
{
	RowVectorXd tsSample, filtered;
	MatrixXd newSample;

	if (!TangentSpace(sample, tsSample, m_Ref)) { return false; }		// Transform to the Tangent Space
	if (!FgDAApply(tsSample, filtered, m_Weight)) { return false; }		// Apply Filter
	if (!UnTangentSpace(filtered, newSample, m_Ref)) { return false; }	// Return to Matrix Space
	return CMatrixClassifierMDM::classify(newSample, classid, distance, probability);
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDM::saveXML(const string& filename)
{
	XMLDocument xmlDoc;
	// Create Root
	XMLNode* root = xmlDoc.NewElement("Classifier");				// Create root node
	xmlDoc.InsertFirstChild(root);									// Add root to XML

	// Write Header
	XMLElement* data = xmlDoc.NewElement("Classifier-data");		// Create data node
	if (!saveHeaderAttribute(data)) { return false; }				// Save Header attribute

	// Write Reference
	XMLElement* reference = xmlDoc.NewElement("Reference");			// Create Reference node
	if (!saveMatrix(reference, m_Ref)) { return false; }			// Save class
	data->InsertEndChild(reference);								// Add class node to data node

	// Write Weight
	XMLElement* weight = xmlDoc.NewElement("Weight");				// Create Reference node
	if (!saveMatrix(weight, m_Weight)) { return false; }			// Save class
	data->InsertEndChild(weight);									// Add class node to data node

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
bool CMatrixClassifierFgMDM::loadXML(const string& filename)
{
	// Load File
	XMLDocument xmlDoc;
	if (xmlDoc.LoadFile(filename.c_str()) != 0) { return false; }	// Check File Exist and Loading

	// Load Root
	XMLNode* root = xmlDoc.FirstChild();							// Get Root Node
	if (root == nullptr) { return false; }							// Check Root Node Exist

	// Load Header
	XMLElement* data = root->FirstChildElement("Classifier-data");	// Get Data Node
	if (!loadHeaderAttribute(data)) { return false; }				// Load Header attribute

	// Load Reference
	XMLElement* ref = data->FirstChildElement("Reference");			// Get Reference Node
	if (!loadMatrix(ref, m_Ref)) { return false; }					// Load Reference Matrix

	// Load Weight
	XMLElement* weight = data->FirstChildElement("Weight");			// Get Weight Node
	if (!loadMatrix(weight, m_Weight)) { return false; }			// Load Weight Matrix

	// Load Class
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
bool CMatrixClassifierFgMDM::saveHeaderAttribute(XMLElement* element) const
{
	element->SetAttribute("type", "FgMDM");								// Set attribute classifier type
	element->SetAttribute("class-count", int(m_ClassCount));			// Set attribute class count
	element->SetAttribute("metric", MetricToString(m_Metric).c_str());	// Set attribute metric
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDM::loadHeaderAttribute(XMLElement* element)
{
	if (element == nullptr) { return false; }						// Check if Node Exist
	const string classifierType = element->Attribute("type");		// Get type
	if (classifierType != "FgMDM") { return false; }				// Check Type
	setClassCount(element->IntAttribute("class-count"));			// Update Number of class
	m_Metric = StringToMetric(element->Attribute("metric"));		// Update Metric
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDM::isEqual(const CMatrixClassifierFgMDM& obj, const double precision) const
{
	if (!CMatrixClassifierMDM::isEqual(obj, precision)) { return false; }	// Compare base members
	if (!AreEquals(m_Ref, obj.m_Ref, precision)) { return false; }			// Compare Reference
	if (!AreEquals(m_Weight, obj.m_Weight, precision)) { return false; }	// Compare Weight
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CMatrixClassifierFgMDM::copy(const CMatrixClassifierFgMDM& obj)
{
	CMatrixClassifierMDM::copy(obj);
	m_Ref = obj.m_Ref;
	m_Weight = obj.m_Weight;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
stringstream CMatrixClassifierFgMDM::print() const
{
	stringstream ss = CMatrixClassifierMDM::print();				// Print base information
	ss << "Reference matrix : " << endl << m_Ref << endl;			// Reference 
	ss << "Weight matrix : " << endl << m_Weight << endl;			// Print Weight
	return ss;
}
///-------------------------------------------------------------------------------------------------
