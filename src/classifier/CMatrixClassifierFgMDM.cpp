#include "CMatrixClassifierFgMDM.hpp"
#include "utils/Mean.hpp"
#include "utils/Basics.hpp"
#include "utils/Featurization.hpp"
#include "utils/Classification.hpp"
#include <iostream>

using namespace std;
using namespace Eigen;
using namespace tinyxml2;

bool CMatrixClassifierFgMDM::computeFgDA(const vector<vector<MatrixXd>>& datasets)
{
	if (datasets.empty()) { return false; }
	// Compute Reference matrix
	const vector<MatrixXd> data = Vector2DTo1D(datasets);		// Append datasets in one vector
	if (!Mean(data, m_Ref, Metric_Euclidian)) { return false; }

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

	// Compute Weight with LSQR Method
	if (!LSQR(ts, m_Weight)) { return false; }

	// Transform weight
	const MatrixXd w = m_Weight, wT = m_Weight.transpose();
	// colPivHouseholderQr().solve(MatrixXd::Identity(nbClass, nbClass)) Compute the pseudo-inverse of a matrix (M * M^(-1) = I)
	m_Weight = (wT * (w * wT).colPivHouseholderQr().solve(MatrixXd::Identity(nbClass, nbClass))) * w;

	return true;
}

bool CMatrixClassifierFgMDM::train(const vector<vector<MatrixXd>>& datasets)
{
	if (!computeFgDA(datasets)) { return false; }
	return true;
}
///-------------------------------------------------------------------------------------------------

bool CMatrixClassifierFgMDM::classify(const MatrixXd& sample, size_t& classid)
{
	(void)sample;
	(void)classid;
	return true;
}
///-------------------------------------------------------------------------------------------------

bool CMatrixClassifierFgMDM::classify(const MatrixXd& sample, size_t& classid, vector<double>& distance, vector<double>& probability)
{
	(void)sample;
	(void)classid;
	(void)distance;
	(void)probability;
	return true;
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
