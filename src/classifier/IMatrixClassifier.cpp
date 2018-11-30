#include "IMatrixClassifier.hpp"

IMatrixClassifier::IMatrixClassifier(const size_t classcount)
{
	IMatrixClassifier::setClassCount(classcount);
}

void IMatrixClassifier::setClassCount(const size_t classcount)
{
	m_ClassCount = classcount;
}
