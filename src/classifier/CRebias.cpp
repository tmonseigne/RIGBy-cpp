#include "CRebias.hpp"
#include "IMatrixClassifier.hpp"
#include "utils/Mean.hpp"
#include "utils/Basics.hpp"
#include "utils/Geodesic.hpp"
#include <unsupported/Eigen/MatrixFunctions> // SQRT of Matrix
#include <iostream>

using namespace std;
using namespace Eigen;

///-------------------------------------------------------------------------------------------------
bool CRebias::computeRebias(const std::vector<std::vector<MatrixXd>>& datasets, const EMetrics metric)
{
	if (!Mean(Vector2DTo1D(datasets), m_Bias, metric)) { return false; }	// Compute Rebias reference
	m_BiasIS = m_Bias.sqrt().inverse();										// Inverse Square root of Rebias matrix => isR
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
	out = m_BiasIS * in * m_BiasIS.transpose();
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CRebias::updateRebias(const Eigen::MatrixXd& sample, const EMetrics metric)
{
	m_NClassify++;													// Update number of classify
	if (m_NClassify == 1) { m_Bias = sample; }						// At the first pass we reinitialize the Rebias
	else { Geodesic(m_Bias, sample, m_Bias, metric, 1.0 / m_NClassify); }
	m_BiasIS = m_Bias.sqrt().inverse();								// Inverse Square root of Rebias matrix => isR
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CRebias::isEqual(const CRebias& obj, const double precision) const
{
	return AreEquals(m_Bias, obj.m_Bias, precision) && m_NClassify == obj.m_NClassify;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CRebias::copy(const CRebias& obj)
{
	m_Bias = obj.m_Bias;
	m_NClassify = obj.m_NClassify;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
std::stringstream CRebias::print() const
{
	stringstream ss;
	ss << "Number of Classification : " << m_NClassify << endl;
	ss << "REBIAS Matrix : ";
	if (m_Bias.size() != 0) { ss << endl << m_Bias.format(MATRIX_FORMAT) << endl; }
	else { ss << "Not Computed" << endl; }
	return ss;
}
///-------------------------------------------------------------------------------------------------


