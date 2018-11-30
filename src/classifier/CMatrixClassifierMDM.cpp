#include "CMatrixClassifierMDM.hpp"
#include "utils/Mean.hpp"
#include "utils/Distance.hpp"
#include "utils/Basics.hpp"
#include <iostream>

using namespace std;
using namespace Eigen;

CMatrixClassifierMDM::CMatrixClassifierMDM()
{
	m_Means.resize(m_ClassCount);
}

CMatrixClassifierMDM::CMatrixClassifierMDM(const size_t classcount, const EMetrics metric)
{
	CMatrixClassifierMDM::setClassCount(classcount);
	m_Metric = metric;
}
///-------------------------------------------------------------------------------------------------

void CMatrixClassifierMDM::setClassCount(const size_t classcount)
{
	if (m_ClassCount != classcount)
	{
		IMatrixClassifier::setClassCount(classcount);
		m_Means.resize(m_ClassCount);
	}
}

bool CMatrixClassifierMDM::train(const vector<vector<MatrixXd>>& datasets)
{
	setClassCount(datasets.size());										// Change the number of class if needed
	for (size_t i = 0; i < m_ClassCount; ++i)
	{
		if (!Mean(datasets[i], m_Means[i], m_Metric)) { return false; }	// Compute the mean of each class
	}
	return true;
}
///-------------------------------------------------------------------------------------------------

bool CMatrixClassifierMDM::classify(const MatrixXd& sample, uint32_t& classid)
{
	std::vector<double> distance, probability;
	return classify(sample, classid, distance, probability);
}
///-------------------------------------------------------------------------------------------------


bool CMatrixClassifierMDM::classify(const MatrixXd& sample, uint32_t& classid, vector<double>& distance, vector<double>& probability)
{
	if (!isSquare(sample)) { return false; }
	double distMin = std::numeric_limits<double>::max();
	// Compute Distance
	distance.resize(m_ClassCount);
	for (size_t i = 0; i < m_ClassCount; ++i)
	{
		distance[i] = Distance(sample, m_Means[i], m_Metric);
		if (distMin > distance[i])
		{
			classid = uint32_t(i);
			distMin = distance[i];
		}
	}

	// Compute Probability (personnal method)
	probability.resize(m_ClassCount);
	double sumProbability = 0.0;
	for (size_t i = 0; i < m_ClassCount; ++i)
	{
		probability[i] = distMin / distance[i];
		sumProbability += probability[i];
	}

	for (auto& p : probability)
	{
		p /= sumProbability;
	}
	return true;
}

bool CMatrixClassifierMDM::operator==(const CMatrixClassifierMDM& obj) const
{
	if (m_Metric != obj.m_Metric || m_ClassCount != obj.m_ClassCount) { return false; }
	for (size_t i = 0; i < m_ClassCount; ++i)
	{
		if (m_Means[i] != obj.m_Means[i]) { return false; }
	}
	return true;
}

bool CMatrixClassifierMDM::operator!=(const CMatrixClassifierMDM& obj) const
{
	return !(*this == obj);
}
///-------------------------------------------------------------------------------------------------
ostream& operator<<(ostream& os, const CMatrixClassifierMDM& obj)
{
	os << "Metric : " << MetricToString(obj.m_Metric) << endl
		<< "Nb of Class : " << obj.m_ClassCount << endl;
	for (size_t i = 0; i < obj.m_ClassCount; ++i)
	{
		os << "Mean Class " << i << " : ";
		if (obj.m_Means[i].size() != 0)
		{
			os << endl << obj.m_Means[i] << endl;
		}
		else
		{
			os << "Not Compute" << endl;
		}
	}
	return os;
}
