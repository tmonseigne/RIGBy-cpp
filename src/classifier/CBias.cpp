#include "geometry/classifier/CBias.hpp"
#include "geometry/classifier/IMatrixClassifier.hpp"
#include "geometry/Mean.hpp"
#include "geometry/Basics.hpp"
#include "geometry/Geodesic.hpp"
#include <unsupported/Eigen/MatrixFunctions> // SQRT of Matrix
#include <iostream>

namespace Geometry {

///-------------------------------------------------------------------------------------------------
bool CBias::computeBias(const std::vector<std::vector<Eigen::MatrixXd>>& datasets, const EMetric metric) { return computeBias(Vector2DTo1D(datasets), metric); }
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CBias::computeBias(const std::vector<Eigen::MatrixXd>& datasets, const EMetric metric)
{
	if (!Mean(datasets, m_bias, metric)) { return false; }	// Compute Bias reference
	m_biasIS = m_bias.sqrt().inverse();						// Inverse Square root of Bias matrix => isR
	m_n      = 0;
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CBias::applyBias(const std::vector<std::vector<Eigen::MatrixXd>>& in, std::vector<std::vector<Eigen::MatrixXd>>& out)
{
	const size_t n = in.size();
	out.resize(n);
	for (size_t i = 0; i < n; ++i) { applyBias(in[i], out[i]); }
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CBias::applyBias(const std::vector<Eigen::MatrixXd>& in, std::vector<Eigen::MatrixXd>& out)
{
	const size_t n = in.size();
	out.resize(n);
	for (size_t i = 0; i < n; ++i) { applyBias(in[i], out[i]); }
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CBias::applyBias(const Eigen::MatrixXd& in, Eigen::MatrixXd& out) { out = m_biasIS * in * m_biasIS.transpose(); }
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CBias::updateBias(const Eigen::MatrixXd& sample, const EMetric metric)
{
	m_n++;													// Update number of classify
	if (m_n == 1) { m_bias = sample; }						// At the first pass we reinitialize the Bias
	else { Geodesic(m_bias, sample, m_bias, metric, 1.0 / m_n); }
	m_biasIS = m_bias.sqrt().inverse();						// Inverse Square root of Bias matrix => isR
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CBias::setBias(const Eigen::MatrixXd& bias)
{
	m_bias   = bias;
	m_biasIS = m_bias.sqrt().inverse();
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CBias::saveXML(const std::string& filename) const
{
	tinyxml2::XMLDocument xmlDoc;
	// Create Root
	tinyxml2::XMLNode* root = xmlDoc.NewElement("Bias");	// Create root node
	xmlDoc.InsertFirstChild(root);							// Add root to XML

	tinyxml2::XMLElement* data = xmlDoc.NewElement("Bias-data");	// Create data node
	if (!saveAdditional(xmlDoc, data)) { return false; }	// Save Optionnal Informations

	root->InsertEndChild(data);								// Add data to root
	return xmlDoc.SaveFile(filename.c_str()) == 0;			// save XML (if != 0 it means error)
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CBias::loadXML(const std::string& filename)
{
	// Load File
	tinyxml2::XMLDocument xmlDoc;
	if (xmlDoc.LoadFile(filename.c_str()) != 0) { return false; }	// Check File Exist and Loading

	// Load Root
	tinyxml2::XMLNode* root = xmlDoc.FirstChild();			// Get Root Node
	if (root == nullptr) { return false; }					// Check Root Node Exist

	// Load Data
	tinyxml2::XMLElement* data = root->FirstChildElement("Bias-data");	// Get Data Node
	if (!loadAdditional(data)) { return false; }			// Load Optionnal Informations
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CBias::saveAdditional(tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* data) const
{
	tinyxml2::XMLElement* bias = doc.NewElement("Bias");	// Create Bias node
	bias->SetAttribute("n", int(m_n));						// Set attribute class number of trials
	if (!IMatrixClassifier::saveMatrix(bias, m_bias)) { return false; }	// Save class
	data->InsertEndChild(bias);								// Add class node to data node
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CBias::loadAdditional(tinyxml2::XMLElement* data)
{
	tinyxml2::XMLElement* bias = data->FirstChildElement("Bias");	// Get LDA Weight Node
	m_n                        = bias->IntAttribute("n");			// Get the number of Trials for this class
	if (!IMatrixClassifier::loadMatrix(bias, m_bias)) { return false; }	// Load Reference Matrix
	m_biasIS = m_bias.sqrt().inverse();
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CBias::isEqual(const CBias& obj, const double precision) const { return AreEquals(m_bias, obj.m_bias, precision) && m_n == obj.m_n; }
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CBias::copy(const CBias& obj)
{
	m_bias   = obj.m_bias;
	m_biasIS = obj.m_biasIS;
	m_n      = obj.m_n;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
std::stringstream CBias::print() const
{
	std::stringstream ss;
	ss << "Number of Classification : " << m_n << std::endl;
	ss << "Bias Matrix : ";
	if (m_bias.size() != 0) { ss << std::endl << m_bias.format(MATRIX_FORMAT) << std::endl; }
	else { ss << "Not Computed" << std::endl; }
	return ss;
}
///-------------------------------------------------------------------------------------------------

}  // namespace Geometry
