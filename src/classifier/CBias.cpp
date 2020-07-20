#include "CBias.hpp"
#include "IMatrixClassifier.hpp"
#include "utils/Mean.hpp"
#include "utils/Basics.hpp"
#include "utils/Geodesic.hpp"
#include <unsupported/Eigen/MatrixFunctions> // SQRT of Matrix
#include <iostream>

using namespace std;
using namespace Eigen;
using namespace tinyxml2;

///-------------------------------------------------------------------------------------------------
bool CBias::computeBias(const std::vector<std::vector<MatrixXd>>& datasets, const EMetric metric) { return computeBias(Vector2DTo1D(datasets), metric); }
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CBias::computeBias(const std::vector<MatrixXd>& datasets, const EMetric metric)
{
	if (!Mean(datasets, m_bias, metric)) { return false; }					// Compute Bias reference
	m_biasIS = m_bias.sqrt().inverse();										// Inverse Square root of Bias matrix => isR
	m_N      = 0;
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CBias::applyBias(const std::vector<std::vector<MatrixXd>>& in, std::vector<std::vector<MatrixXd>>& out)
{
	const size_t n = in.size();
	out.resize(n);
	for (size_t i = 0; i < n; ++i) { applyBias(in[i], out[i]); }
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CBias::applyBias(const std::vector<MatrixXd>& in, std::vector<MatrixXd>& out)
{
	const size_t n = in.size();
	out.resize(n);
	for (size_t i = 0; i < n; ++i) { applyBias(in[i], out[i]); }
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CBias::applyBias(const MatrixXd& in, MatrixXd& out) { out = m_biasIS * in * m_biasIS.transpose(); }
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CBias::updateBias(const MatrixXd& sample, const EMetric metric)
{
	m_N++;													// Update number of classify
	if (m_N == 1) { m_bias = sample; }						// At the first pass we reinitialize the Bias
	else { Geodesic(m_bias, sample, m_bias, metric, 1.0 / m_N); }
	m_biasIS = m_bias.sqrt().inverse();								// Inverse Square root of Bias matrix => isR
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CBias::setBias(const MatrixXd& bias)
{
	m_bias   = bias;
	m_biasIS = m_bias.sqrt().inverse();
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CBias::saveXML(const std::string& filename) const
{
	XMLDocument xmlDoc;
	// Create Root
	XMLNode* root = xmlDoc.NewElement("Bias");				// Create root node
	xmlDoc.InsertFirstChild(root);								// Add root to XML

	XMLElement* data = xmlDoc.NewElement("Bias-data");		// Create data node
	if (!saveAdditional(xmlDoc, data)) { return false; }		// Save Optionnal Informations

	root->InsertEndChild(data);									// Add data to root
	return xmlDoc.SaveFile(filename.c_str()) == 0;				// save XML (if != 0 it means error)
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CBias::loadXML(const std::string& filename)
{
	// Load File
	XMLDocument xmlDoc;
	if (xmlDoc.LoadFile(filename.c_str()) != 0) { return false; }	// Check File Exist and Loading

	// Load Root
	XMLNode* root = xmlDoc.FirstChild();							// Get Root Node
	if (root == nullptr) { return false; }							// Check Root Node Exist

	// Load Data
	XMLElement* data = root->FirstChildElement("Bias-data");		// Get Data Node
	if (!loadAdditional(data)) { return false; }					// Load Optionnal Informations
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CBias::saveAdditional(XMLDocument& doc, XMLElement* data) const
{
	XMLElement* bias = doc.NewElement("Bias");							// Create Bias node
	bias->SetAttribute("n", int(m_N));									// Set attribute class number of trials
	if (!IMatrixClassifier::saveMatrix(bias, m_bias)) { return false; }	// Save class
	data->InsertEndChild(bias);											// Add class node to data node
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CBias::loadAdditional(XMLElement* data)
{
	XMLElement* bias = data->FirstChildElement("Bias");					// Get LDA Weight Node
	m_N              = bias->IntAttribute("n");										// Get the number of Trials for this class
	if (!IMatrixClassifier::loadMatrix(bias, m_bias)) { return false; }	// Load Reference Matrix
	m_biasIS = m_bias.sqrt().inverse();
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CBias::isEqual(const CBias& obj, const double precision) const { return AreEquals(m_bias, obj.m_bias, precision) && m_N == obj.m_N; }
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CBias::copy(const CBias& obj)
{
	m_bias = obj.m_bias;
	m_N    = obj.m_N;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
std::stringstream CBias::print() const
{
	stringstream ss;
	ss << "Number of Classification : " << m_N << endl;
	ss << "Bias Matrix : ";
	if (m_bias.size() != 0) { ss << endl << m_bias.format(MATRIX_FORMAT) << endl; }
	else { ss << "Not Computed" << endl; }
	return ss;
}
///-------------------------------------------------------------------------------------------------
