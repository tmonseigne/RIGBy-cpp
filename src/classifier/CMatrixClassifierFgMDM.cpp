#include "CMatrixClassifierFgMDM.hpp"
#include "utils/Mean.hpp"
#include "utils/Basics.hpp"
#include "utils/Featurization.hpp"
#include "utils/Classification.hpp"

using namespace std;
using namespace Eigen;
using namespace tinyxml2;

bool CMatrixClassifierFgMDM::train(const vector<vector<MatrixXd>>& datasets)
{
	if (datasets.empty()) { return false; }
	// Compute Reference matrix
	const vector<MatrixXd> data = Vector2DTo1D(datasets);		// Append datasets in one vector
	if (!Mean(data, m_Ref, Metric_Riemann)) { return false; }

	// Transform to the Tangent Space
	const size_t nbClass = datasets.size();
	vector<vector<RowVectorXd>> ts(nbClass);
	for (size_t k = 0; k < nbClass; ++k)
	{
		const size_t nbTrials = datasets[k].size();
		ts[k].resize(nbTrials);
		for (size_t i = 0; i < nbTrials; ++i)
		{
			if (!TangentSpace(datasets[k][i], ts[k][i], m_Ref)) { return false; }
		}
	}

	// Compute FgDA Weight
	if (!(FgDACompute(ts, m_Weight))) { return false; }


	// Convert Dataset
	vector<vector<MatrixXd>> newDatasets(nbClass);
	vector<vector<RowVectorXd>> filtered(nbClass);
	for (size_t k = 0; k < nbClass; ++k)
	{
		const size_t nbTrials = datasets[k].size();
		newDatasets[k].resize(nbTrials);
		filtered[k].resize(nbTrials);
		for (size_t i = 0; i < nbTrials; ++i)
		{
			if (!FgDAApply(ts[k][i], filtered[k][i], m_Weight)) { return false; }				// Apply Filter
			if (!UnTangentSpace(filtered[k][i], newDatasets[k][i], m_Ref)) { return false; }	// Return to Matrix Space
		}
	}

	return CMatrixClassifierMDM::train(newDatasets);											// Train MDM
}
///-------------------------------------------------------------------------------------------------

bool CMatrixClassifierFgMDM::classify(const MatrixXd& sample, size_t& classid)
{
	std::vector<double> distance, probability;
	return classify(sample, classid, distance, probability);
}
///-------------------------------------------------------------------------------------------------

bool CMatrixClassifierFgMDM::classify(const MatrixXd& sample, size_t& classid, vector<double>& distance, vector<double>& probability)
{
	RowVectorXd ts, filtered;
	MatrixXd newSample;
	
	if (!TangentSpace(sample, ts, m_Ref)) { return false; }				// Transform to the Tangent Space
	if (!FgDAApply(ts, filtered, m_Weight)) { return false; }			// Apply Filter
	if (!UnTangentSpace(filtered, newSample, m_Ref)) { return false; }	// Return to Matrix Space
	return CMatrixClassifierMDM::classify(newSample, classid, distance, probability);
}
///-------------------------------------------------------------------------------------------------

bool CMatrixClassifierFgMDM::saveXML(const string& filename)
{
	(void)filename;
	return true;
}
///-------------------------------------------------------------------------------------------------

bool CMatrixClassifierFgMDM::loadXML(const string& filename)
{
	(void)filename;
	return true;
}
///-------------------------------------------------------------------------------------------------

bool CMatrixClassifierFgMDM::saveHeaderAttribute(XMLElement* element) const
{
	(void)element;
	return true;
}
///-------------------------------------------------------------------------------------------------

bool CMatrixClassifierFgMDM::loadHeaderAttribute(XMLElement* element)
{
	(void)element;
	return true;
}
///-------------------------------------------------------------------------------------------------

bool CMatrixClassifierFgMDM::saveClass(XMLElement* element, const size_t index) const
{
	(void)element;
	(void)index;
	return true;
}
///-------------------------------------------------------------------------------------------------

bool CMatrixClassifierFgMDM::loadClass(XMLElement* element, const size_t index)
{
	(void)element;
	(void)index;
	return true;
}
///-------------------------------------------------------------------------------------------------

bool CMatrixClassifierFgMDM::operator==(const CMatrixClassifierFgMDM& obj) const
{
	(void)obj;
	return true;
}
///-------------------------------------------------------------------------------------------------

bool CMatrixClassifierFgMDM::operator!=(const CMatrixClassifierFgMDM& obj) const
{
	(void)obj;
	return true;
}
///-------------------------------------------------------------------------------------------------


stringstream CMatrixClassifierFgMDM::print() const
{
	stringstream ss;
	ss << "Nb of Class : " << m_ClassCount << endl;
	return ss;
}
///-------------------------------------------------------------------------------------------------

ostream& operator<<(ostream& os, const CMatrixClassifierFgMDM& obj)
{
	os << obj.print().str();
	return os;
}
///-------------------------------------------------------------------------------------------------
