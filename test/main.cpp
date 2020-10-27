#include "gtest/gtest.h"

// ReSharper disable CppUnusedIncludeDirective
#include "test_Basics.hpp"
#include "test_Covariance.hpp"
#include "test_Mean.hpp"
#include "test_Median.hpp"
#include "test_Misc.hpp"
#include "test_Distance.hpp"
#include "test_Geodesics.hpp"
#include "test_Featurization.hpp"
#include "test_Classifier.hpp"
#include "test_MatrixClassifier.hpp"
#include "test_ASR.hpp"
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
		const Geometry::CMatrixClassifierMDM dumC;
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
