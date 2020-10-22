#include "geometry/Geodesic.hpp"
#include "geometry/Basics.hpp"
#include <unsupported/Eigen/MatrixFunctions>

namespace Geometry {

//---------------------------------------------------------------------------------------------------
bool Geodesic(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, Eigen::MatrixXd& g, const EMetric metric, const double alpha)
{
	if (!HaveSameSize(a, b)) { return false; }						// Verification same size
	if (!IsSquare(a)) { return false; }								// Verification square matrix
	if (!InRange(alpha, 0, 1)) { return false; }					// Verification alpha in [0;1]
	switch (metric)													// Switch metric
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
bool GeodesicRiemann(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, Eigen::MatrixXd& g, const double alpha)
{
	const Eigen::MatrixXd sA = a.sqrt(), isA = sA.inverse();
	g                        = sA * (isA * b * isA).pow(alpha) * sA;
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool GeodesicEuclidian(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, Eigen::MatrixXd& g, const double alpha)
{
	g = (1 - alpha) * a + alpha * b;
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool GeodesicLogEuclidian(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, Eigen::MatrixXd& g, const double alpha)
{
	g = ((1 - alpha) * a.log() + alpha * b.log()).exp();
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool GeodesicIdentity(const Eigen::MatrixXd& a, const Eigen::MatrixXd& /*b*/, Eigen::MatrixXd& g, const double /*alpha*/)
{
	g = Eigen::MatrixXd::Identity(a.rows(), a.rows());
	return true;
}
//---------------------------------------------------------------------------------------------------

}  // namespace Geometry
