#include "geometry/Distance.hpp"
#include "geometry/Basics.hpp"
#include <unsupported/Eigen/MatrixFunctions>

namespace Geometry {

//---------------------------------------------------------------------------------------------------
double Distance(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, const EMetric metric)
{
	if (!HaveSameSize(a, b)) { return 0; }
	switch (metric)
	{
		case EMetric::Riemann: return DistanceRiemann(a, b);
		case EMetric::Euclidian: return DistanceEuclidian(a, b);
		case EMetric::LogEuclidian: return DistanceLogEuclidian(a, b);
		case EMetric::LogDet: return DistanceLogDet(a, b);
		case EMetric::Kullback: return DistanceKullbackSym(a, b);
		case EMetric::Wasserstein: return DistanceWasserstein(a, b);
		case EMetric::Identity:
		default: return 1.0;
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
double DistanceRiemann(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b)
{
	const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(a, b);
	const Eigen::ArrayXd result = es.eigenvalues();
	return sqrt(result.log().square().sum());
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
double DistanceEuclidian(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) { return (b - a).norm(); }
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
double DistanceLogEuclidian(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) { return DistanceEuclidian(a.log(), b.log()); }
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
double DistanceLogDet(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b)
{
	return sqrt(log((0.5 * (a + b)).determinant()) - 0.5 * log(a.determinant() * b.determinant()));
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
double DistanceKullback(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b)
{
	return 0.5 * ((b.inverse() * a).trace() - a.rows() + log(b.determinant() / a.determinant()));
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
double DistanceKullbackSym(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) { return DistanceKullback(a, b) + DistanceKullback(b, a); }
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
double DistanceWasserstein(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b)
{
	const Eigen::MatrixXd sB = b.sqrt();
	return sqrt((a + b - 2 * (sB * a * sB).sqrt()).trace());
}
//---------------------------------------------------------------------------------------------------

}  // namespace Geometry
