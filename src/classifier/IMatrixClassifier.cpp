#include "IMatrixClassifier.hpp"
#include <iostream>

using namespace std;
using namespace Eigen;
using namespace tinyxml2;

//***********************	
//***** Constructor *****	
//***********************
///-------------------------------------------------------------------------------------------------
IMatrixClassifier::IMatrixClassifier(const size_t classcount, const EMetrics metric)
{
	IMatrixClassifier::setClassCount(classcount);
	m_Metric = metric;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
IMatrixClassifier::IMatrixClassifier(const IMatrixClassifier& obj)
{
	copy(obj);
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void IMatrixClassifier::setClassCount(const size_t classcount)
{
	m_classCount = classcount;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool IMatrixClassifier::classify(const MatrixXd& sample, size_t& classId, const EAdaptations adaptation, const size_t& realClassId)
{
	vector<double> distance, probability;
	return classify(sample, classId, distance, probability, adaptation, realClassId);
}
///-------------------------------------------------------------------------------------------------

//***********************
//***** XML Manager *****
//***********************
///-------------------------------------------------------------------------------------------------
bool IMatrixClassifier::convertMatrixToXMLFormat(const MatrixXd& in, stringstream& out)
{
	const IOFormat fmt(FullPrecision, 0, " ", "\n", "", "", "", "");
	out << in.format(fmt);
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool IMatrixClassifier::convertXMLFormatToMatrix(stringstream& in, MatrixXd& out, const size_t rows, const size_t cols)
{
	out = MatrixXd::Identity(rows, cols);				// Init With Identity Matrix (in case of)
	for (size_t i = 0; i < rows; ++i)					// Fill Matrix
	{
		for (size_t j = 0; j < cols; ++j) { in >> out(i, j); }
	}
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool IMatrixClassifier::saveMatrix(XMLElement* element, const MatrixXd& matrix)
{
	element->SetAttribute("size", int(matrix.rows()));	// Set Matrix size NxN
	stringstream ss;
	convertMatrixToXMLFormat(matrix, ss);
	element->SetText(ss.str().c_str());					// Write Means Value
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool IMatrixClassifier::loadMatrix(XMLElement* element, MatrixXd& matrix)
{
	const size_t size = element->IntAttribute("size");	// Get number of row/col
	stringstream ss(element->GetText());				// String stream to parse Matrix value
	convertXMLFormatToMatrix(ss, matrix, size, size);
	return true;
}
///-------------------------------------------------------------------------------------------------

//*****************************
//***** Override Operator *****
//*****************************
///-------------------------------------------------------------------------------------------------
bool IMatrixClassifier::isEqual(const IMatrixClassifier& obj, const double /*precision*/) const
{
	return m_Metric == obj.m_Metric && m_classCount == obj.m_classCount;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void IMatrixClassifier::copy(const IMatrixClassifier& obj)
{
	m_Metric = obj.m_Metric;
	m_classCount = obj.m_classCount;
}
///-------------------------------------------------------------------------------------------------
