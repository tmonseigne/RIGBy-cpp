#include "gtest/gtest.h"

// ReSharper disable CppUnusedIncludeDirective
#include "test_RiemannUtilsBasics.hpp"
//#include "test_RiemannUtilsCovariance.hpp"
#include "test_RiemannUtilsDistance.hpp"
#include "test_RiemannUtilsFeaturization.hpp"
#include "test_RiemannUtilsGeodesics.hpp"
//#include "test_RiemannUtilsMean.hpp"
#include "test_RiemannClassif.hpp"
// ReSharper restore CppUnusedIncludeDirective

int main(int argc, char** argv)
{
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
