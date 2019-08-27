#include "CRebias.hpp"
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
bool CRebias::computeRebias(const std::vector<std::vector<MatrixXd>>& datasets, const EMetrics metric)
{
	if (!Mean(Vector2DTo1D(datasets), m_bias, metric)) { return false; }	// Compute Rebias reference
	m_biasIS = m_bias.sqrt().inverse();										// Inverse Square root of Rebias matrix => isR
	m_NClassify = 0;
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CRebias::applyRebias(const std::vector<std::vector<MatrixXd>>& in, std::vector<std::vector<MatrixXd>>& out)
{
	const size_t n = in.size();
	out.resize(n);
	for (size_t i = 0; i < n; ++i) { applyRebias(in[i], out[i]); }
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CRebias::applyRebias(const std::vector<MatrixXd>& in, std::vector<MatrixXd>& out)
{
	const size_t n = in.size();
	out.resize(n);
	for (size_t i = 0; i < n; ++i) { applyRebias(in[i], out[i]); }
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CRebias::applyRebias(const MatrixXd& in, MatrixXd& out)
{
	out = m_biasIS * in * m_biasIS.transpose();
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CRebias::updateRebias(const MatrixXd& sample, const EMetrics metric)
{
	m_NClassify++;													// Update number of classify
	if (m_NClassify == 1) { m_bias = sample; }						// At the first pass we reinitialize the Rebias
	else { Geodesic(m_bias, sample, m_bias, metric, 1.0 / m_NClassify); }
	m_biasIS = m_bias.sqrt().inverse();								// Inverse Square root of Rebias matrix => isR
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CRebias::setBias(MatrixXd& bias)
{
	m_bias   = bias;
	m_biasIS = m_bias.sqrt().inverse();
}

///-------------------------------------------------------------------------------------------------
bool CRebias::save(XMLDocument& doc, XMLElement* data) const
{
	XMLElement* rebias = doc.NewElement("Rebias");							// Create Rebias node
	rebias->SetAttribute("nb-classify", int(m_NClassify));					// Set attribute class number of trials
	if (!IMatrixClassifier::saveMatrix(rebias, m_bias)) { return false; }	// Save class
	data->InsertEndChild(rebias);											// Add class node to data node
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CRebias::load(XMLElement* data)
{
	XMLElement* rebias = data->FirstChildElement("Rebias");					// Get LDA Weight Node
	m_NClassify = rebias->IntAttribute("nb-classify");						// Get the number of Trials for this class
	if (!IMatrixClassifier::loadMatrix(rebias, m_bias)) { return false; }	// Load Reference Matrix
	m_biasIS = m_bias.sqrt().inverse();
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CRebias::isEqual(const CRebias& obj, const double precision) const
{
	return AreEquals(m_bias, obj.m_bias, precision) && m_NClassify == obj.m_NClassify;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CRebias::copy(const CRebias& obj)
{
	m_bias = obj.m_bias;
	m_NClassify = obj.m_NClassify;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
std::stringstream CRebias::print() const
{
	stringstream ss;
	ss << "Number of Classification : " << m_NClassify << endl;
	ss << "REBIAS Matrix : ";
	if (m_bias.size() != 0) { ss << endl << m_bias.format(MATRIX_FORMAT) << endl; }
	else { ss << "Not Computed" << endl; }
	return ss;
}
///-------------------------------------------------------------------------------------------------


