#include "geometry/classifier/CMatrixClassifierFgMDMRT.hpp"
#include "geometry/Mean.hpp"
#include "geometry/Basics.hpp"
#include "geometry/Featurization.hpp"
#include "geometry/Classification.hpp"
#include <iostream>

namespace Geometry {

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDMRT::train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets)
{
	if (datasets.empty()) { return false; }
	if (!Mean(Vector2DTo1D(datasets), m_ref, EMetric::Riemann)) { return false; }	// Compute Reference matrix

	// Transform to the Tangent Space
	const size_t nbClass = datasets.size();
	std::vector<std::vector<Eigen::RowVectorXd>> tsSample(nbClass);
	for (size_t k = 0; k < nbClass; ++k)
	{
		const size_t nbTrials = datasets[k].size();
		tsSample[k].resize(nbTrials);
		for (size_t i = 0; i < nbTrials; ++i) { if (!TangentSpace(datasets[k][i], tsSample[k][i], m_ref)) { return false; } }
	}

	// Compute FgDA Weight
	if (!FgDACompute(tsSample, m_weight)) { return false; }

	// Convert Datasets
	std::vector<std::vector<Eigen::MatrixXd>> newDatasets(nbClass);
	std::vector<std::vector<Eigen::RowVectorXd>> filtered(nbClass);
	for (size_t k = 0; k < nbClass; ++k)
	{
		const size_t nbTrials = datasets[k].size();
		newDatasets[k].resize(nbTrials);
		filtered[k].resize(nbTrials);
		for (size_t i = 0; i < nbTrials; ++i)
		{
			if (!FgDAApply(tsSample[k][i], filtered[k][i], m_weight)) { return false; }			// Apply Filter
			if (!UnTangentSpace(filtered[k][i], newDatasets[k][i], m_ref)) { return false; }	// Return to Matrix Space
		}
	}

	return CMatrixClassifierMDM::train(newDatasets);					// Train MDM
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDMRT::classify(const Eigen::MatrixXd& sample, size_t& classId, std::vector<double>& distance,
										std::vector<double>& probability, const EAdaptations adaptation, const size_t& realClassId)
{
	Eigen::RowVectorXd tsSample, filtered;
	Eigen::MatrixXd newSample;

	if (!TangentSpace(sample, tsSample, m_ref)) { return false; }		// Transform to the Tangent Space
	if (!FgDAApply(tsSample, filtered, m_weight)) { return false; }		// Apply Filter
	if (!UnTangentSpace(filtered, newSample, m_ref)) { return false; }	// Return to Matrix Space
	return CMatrixClassifierMDM::classify(newSample, classId, distance, probability, adaptation, realClassId);
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDMRT::isEqual(const CMatrixClassifierFgMDMRT& obj, const double precision) const
{
	if (!CMatrixClassifierMDM::isEqual(obj, precision)) { return false; }	// Compare base members
	if (!AreEquals(m_ref, obj.m_ref, precision)) { return false; }			// Compare Reference
	if (!AreEquals(m_weight, obj.m_weight, precision)) { return false; }	// Compare Weight
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CMatrixClassifierFgMDMRT::copy(const CMatrixClassifierFgMDMRT& obj)
{
	CMatrixClassifierMDM::copy(obj);
	m_ref    = obj.m_ref;
	m_weight = obj.m_weight;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDMRT::saveAdditional(tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* data) const
{
	// Save Reference
	tinyxml2::XMLElement* reference = doc.NewElement("Reference");	// Create Reference node
	if (!saveMatrix(reference, m_ref)) { return false; }		// Save class
	data->InsertEndChild(reference);							// Add class node to data node

	// Save Weight
	tinyxml2::XMLElement* weight = doc.NewElement("Weight");				// Create LDA Weight node
	if (!saveMatrix(weight, m_weight)) { return false; }		// Save class
	data->InsertEndChild(weight);								// Add class node to data node

	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDMRT::loadAdditional(tinyxml2::XMLElement* data)
{
	// Load Reference
	tinyxml2::XMLElement* ref = data->FirstChildElement("Reference");		// Get Reference Node
	if (!loadMatrix(ref, m_ref)) { return false; }				// Load Reference Matrix

	// Load Weight
	tinyxml2::XMLElement* weight = data->FirstChildElement("Weight");		// Get LDA Weight Node
	return loadMatrix(weight, m_weight);						// Load LDA Weight Matrix
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
std::stringstream CMatrixClassifierFgMDMRT::printAdditional() const
{
	std::stringstream ss;
	ss << "Reference matrix : " << std::endl << m_ref.format(MATRIX_FORMAT) << std::endl;		// Reference 
	ss << "Weight matrix : " << std::endl << m_weight.format(MATRIX_FORMAT) << std::endl;		// Print Weight
	return ss;
}
///-------------------------------------------------------------------------------------------------

}  // namespace Geometry
