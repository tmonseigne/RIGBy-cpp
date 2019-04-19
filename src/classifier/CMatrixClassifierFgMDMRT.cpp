#include "CMatrixClassifierFgMDMRT.hpp"
#include "utils/Mean.hpp"
#include "utils/Basics.hpp"
#include "utils/Featurization.hpp"
#include "utils/Classification.hpp"

using namespace std;
using namespace Eigen;
using namespace tinyxml2;

///-------------------------------------------------------------------------------------------------
bool CMatrixClassifierFgMDMRT::classify(const MatrixXd& sample, size_t& classId, std::vector<double>& distance,
										std::vector<double>& probability, const EAdaptations adaptation, const size_t& realClassId)
{
	RowVectorXd tsSample, filtered;
	MatrixXd newSample;

	if (!TangentSpace(sample, tsSample, m_Ref)) { return false; }		// Transform to the Tangent Space
	if (!FgDAApply(tsSample, filtered, m_Weight)) { return false; }		// Apply Filter
	if (!UnTangentSpace(filtered, newSample, m_Ref)) { return false; }	// Return to Matrix Space

	return CMatrixClassifierMDM::classify(newSample, classId, distance, probability, adaptation, realClassId);
}
///-------------------------------------------------------------------------------------------------
