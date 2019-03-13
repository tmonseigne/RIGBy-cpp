#include "CMatrixClassifierFgMDM.hpp"
#include "utils/Mean.hpp"
#include "utils/Basics.hpp"
#include "utils/Featurization.hpp"
#include "utils/Classification.hpp"
#include "utils/Geodesic.hpp"

using namespace std;
using namespace Eigen;
using namespace tinyxml2;


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
bool CMatrixClassifierFgMDM::classify(const MatrixXd& sample, size_t& classId, std::vector<double>& distance, 
									  std::vector<double>& probability, const EAdaptations adaptation, const size_t& realClassId)
{
	RowVectorXd tsSample, filtered;
	MatrixXd newSample;

	if (!TangentSpace(sample, tsSample, m_Ref)) { return false; }		// Transform to the Tangent Space
	if (!FgDAApply(tsSample, filtered, m_Weight)) { return false; }		// Apply Filter
	if (!UnTangentSpace(filtered, newSample, m_Ref)) { return false; }	// Return to Matrix Space
	CMatrixClassifierMDM::classify(newSample, classId, distance, probability/*, Adaptation_None, std::numeric_limits<std::size_t>::max()*/);

	// Adaptation
	if (adaptation == Adaptation_None) { return true; }
	// Get class id for adaptation and increase number of trials, expected if supervised, predicted if unsupervised
	const size_t id = adaptation == Adaptation_Supervised ? realClassId : classId;
	if (id >= m_classCount) { return false; }	// Check id (if supervised and bad input)
	m_NbTrials[id]++;							// Update number of trials for the class id
	return Geodesic(m_Means[id], sample, m_Means[id], m_Metric, 1.0 / m_NbTrials[id]);
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
bool CMatrixClassifierFgMDM::saveAdditional(XMLDocument& doc, XMLElement* data) const
{
	// Save Reference
	XMLElement* reference = doc.NewElement("Reference");		// Create Reference node
	if (!saveMatrix(reference, m_Ref)) { return false; }		// Save class
	data->InsertEndChild(reference);							// Add class node to data node

	// Save Weight
	XMLElement* weight = doc.NewElement("Weight");				// Create Reference node
	if (!saveMatrix(weight, m_Weight)) { return false; }		// Save class
	data->InsertEndChild(weight);								// Add class node to data node

	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDM::loadAdditional(XMLDocument& /*doc*/, XMLElement* data)
{
	// Load Reference
	XMLElement* ref = data->FirstChildElement("Reference");		// Get Reference Node
	if (!loadMatrix(ref, m_Ref)) { return false; }				// Load Reference Matrix

	// Load Weight
	XMLElement* weight = data->FirstChildElement("Weight");		// Get Weight Node
	return loadMatrix(weight, m_Weight);						// Load REBIAS Matrix
}

std::stringstream CMatrixClassifierFgMDM::printAdditional() const
{
	stringstream ss;
	ss << "Reference matrix : " << endl << m_Ref << endl;		// Reference 
	ss << "Weight matrix : " << endl << m_Weight << endl;		// Print Weight
	return ss;

}
///-------------------------------------------------------------------------------------------------
