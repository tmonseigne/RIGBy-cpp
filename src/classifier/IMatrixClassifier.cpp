#include "IMatrixClassifier.hpp"


IMatrixClassifier::IMatrixClassifier(const size_t classcount, const EMetrics metric)
{
	IMatrixClassifier::setClassCount(classcount);
	m_Metric = metric;
}
///-------------------------------------------------------------------------------------------------

void IMatrixClassifier::setClassCount(const size_t classcount)
{
	m_ClassCount = classcount;
}
///-------------------------------------------------------------------------------------------------