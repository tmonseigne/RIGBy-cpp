#include "CMatrixClassifierFgDA.hpp"

using namespace std;


bool CMatrixClassifierFgDA::compute(std::vector<Eigen::MatrixXd>& dataset)
{
	(void)dataset;
	return true;
}
///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgDA::filter(Eigen::MatrixXd& sample)
{
	(void)sample;
	return true;
}
///-------------------------------------------------------------------------------------------------

bool CMatrixClassifierFgDA::operator==(const CMatrixClassifierFgDA& obj) const
{
	(void)obj;
	return true;
}
///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgDA::operator!=(const CMatrixClassifierFgDA& obj) const
{
	(void)obj;
	return true;
}
///-------------------------------------------------------------------------------------------------
std::stringstream CMatrixClassifierFgDA::print() const
{
	stringstream ss;
	ss << "Metric : " << MetricToString(m_Metric) << endl;
	return ss;
}
///-------------------------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& os, const CMatrixClassifierFgDA& obj)
{
	os << obj.print().str();
	return os;
}
///-------------------------------------------------------------------------------------------------
