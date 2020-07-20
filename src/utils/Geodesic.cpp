#include "Geodesic.hpp"
#include "Basics.hpp"
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;

//---------------------------------------------------------------------------------------------------
bool Geodesic(const MatrixXd& a, const MatrixXd& b, MatrixXd& g, const EMetric metric, const double alpha)
{
	if (!HaveSameSize(a, b)) { return false; }							// Verification same size
	if (!IsSquare(a)) { return false; }									// Verification square matrix
	if (!InRange(alpha, 0, 1)) { return false; }						// Verification alpha in [0;1]
	switch (metric)														// Switch metric
	{
		case EMetric::Riemann: return GeodesicRiemann(a, b, g, alpha);
		case EMetric::Euclidian: return GeodesicEuclidian(a, b, g, alpha);
		case EMetric::LogEuclidian: return GeodesicLogEuclidian(a, b, g, alpha);
		case EMetric::Identity:
		default: return GeodesicIdentity(a, b, g, alpha);
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool GeodesicRiemann(const MatrixXd& a, const MatrixXd& b, MatrixXd& g, const double alpha)
{
	const MatrixXd sA = a.sqrt(), isA = sA.inverse();
	g                 = sA * (isA * b * isA).pow(alpha) * sA;
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool GeodesicEuclidian(const MatrixXd& a, const MatrixXd& b, MatrixXd& g, const double alpha)
{
	g = (1 - alpha) * a + alpha * b;
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool GeodesicLogEuclidian(const MatrixXd& a, const MatrixXd& b, MatrixXd& g, const double alpha)
{
	g = ((1 - alpha) * a.log() + alpha * b.log()).exp();
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool GeodesicIdentity(const MatrixXd& a, const MatrixXd& /*b*/, MatrixXd& g, const double /*alpha*/)
{
	g = MatrixXd::Identity(a.rows(), a.rows());
	return true;
}
//---------------------------------------------------------------------------------------------------
