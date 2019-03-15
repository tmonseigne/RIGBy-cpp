#include "gtest/gtest.h"

// ReSharper disable CppUnusedIncludeDirective
//*
#include "test_UtilsBasics.hpp"
#include "test_UtilsCovariance.hpp"
#include "test_UtilsMean.hpp"
#include "test_UtilsDistance.hpp"
#include "test_UtilsGeodesics.hpp"
#include "test_UtilsFeaturization.hpp"
#include "test_UtilsClassifier.hpp"
//*/
#include "test_MatrixClassifier.hpp"
// ReSharper restore CppUnusedIncludeDirective

int main(int argc, char** argv)
{
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
