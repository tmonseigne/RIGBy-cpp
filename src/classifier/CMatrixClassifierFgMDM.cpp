#include "CMatrixClassifierFgMDM.hpp"

using namespace std;
using namespace Eigen;
using namespace tinyxml2;

CMatrixClassifierFgMDM::CMatrixClassifierFgMDM(const size_t classcount)
{
	CMatrixClassifierFgMDM::setClassCount(classcount);
}
///-------------------------------------------------------------------------------------------------

void CMatrixClassifierFgMDM::setClassCount(const size_t classcount)
{
	IMatrixClassifier::setClassCount(classcount);
}
///-------------------------------------------------------------------------------------------------

bool CMatrixClassifierFgMDM::train(const std::vector<std::vector<MatrixXd>>& datasets)
{
	(void)datasets;
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

bool CMatrixClassifierFgMDM::classify(const MatrixXd& sample, size_t& classid, std::vector<double>& distance, std::vector<double>& probability)
{
	(void)sample;
	(void)classid;
	(void)distance;
	(void)probability;
	return true;
}
///-------------------------------------------------------------------------------------------------

bool CMatrixClassifierFgMDM::saveXML(const std::string& filename)
{
	(void)filename;
	return true;
}
///-------------------------------------------------------------------------------------------------

bool CMatrixClassifierFgMDM::loadXML(const std::string& filename)
{
	(void)filename;
	return true;
}

bool CMatrixClassifierFgMDM::saveHeaderAttribute(XMLElement* element) const
{
	(void)element;
	return true;
}

bool CMatrixClassifierFgMDM::loadHeaderAttribute(XMLElement* element)
{
	(void)element;
	return true;
}

bool CMatrixClassifierFgMDM::saveClass(XMLElement* element, const size_t index) const
{
	(void)element;
	(void)index;
	return true;
}

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

std::ostream& operator<<(std::ostream& os, const CMatrixClassifierFgMDM& obj)
{
	os << obj.print().str();
	return os;
}
///-------------------------------------------------------------------------------------------------
