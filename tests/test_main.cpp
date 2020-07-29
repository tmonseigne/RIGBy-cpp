#include "gtest/gtest.h"

// ReSharper disable CppUnusedIncludeDirective
#include "test_UtilsBasics.hpp"
#include "test_UtilsCovariance.hpp"
#include "test_UtilsMean.hpp"
#include "test_UtilsMedian.hpp"
#include "test_UtilsDistance.hpp"
#include "test_UtilsGeodesics.hpp"
#include "test_UtilsFeaturization.hpp"
#include "test_UtilsClassifier.hpp"
#include "test_MatrixClassifier.hpp"
// ReSharper restore CppUnusedIncludeDirective

int main(int argc, char** argv)
{
	try
	{
		//Code coverage tips (this functions are used only if tests failed)
		const size_t dumS           = 0;
		const double dumD           = 0;
		const Eigen::MatrixXd dumM  = Eigen::MatrixXd::Identity(2, 2);
		const std::vector<int> dumV = { 0, 0 };
		const CMatrixClassifierMDM dumC;
		ErrorMsg("", dumS, dumS);
		ErrorMsg("", dumD, dumD);
		ErrorMsg("", dumM, dumM);
		ErrorMsg("", dumV, dumV);
		ErrorMsg("", dumC, dumC);
		testing::InitGoogleTest(&argc, argv);
		return RUN_ALL_TESTS();
	}
	catch (std::exception&) { return 1; }
}
