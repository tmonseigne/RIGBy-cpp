#include "Geodesic.hpp"
#include "Basics.hpp"
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;

//---------------------------------------------------------------------------------------------------
bool Geodesic(const MatrixXd& a, const MatrixXd& b, MatrixXd& g, const EMetrics metric, const double alpha)
{
	if (!haveSameSize(a, b)) { return false; }							// Verification same size
	if (!isSquare(a)) { return false; }									// Verification square matrix
	if (!inRange(alpha, 0, 1)) { return false; }						// Vérification alpha in [0;1]
	switch (metric)														// Switch metric
	{
		case Metric_Riemann: return GeodesicRiemann(a, b, g, alpha);
		case Metric_Euclidian: return GeodesicEuclidian(a, b, g, alpha);
		case Metric_LogEuclidian: return GeodesicLogEuclidian(a, b, g, alpha);
		case Metric_Identity:
		default: return GeodesicIdentity(a, b, g, alpha);
	}
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool GeodesicRiemann(const MatrixXd& a, const MatrixXd& b, MatrixXd& g, const double alpha)
{
	const MatrixXd sA = a.sqrt(),
				   isA = sA.inverse();
	g = sA * (isA * b * isA).pow(alpha) * sA;
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
bool GeodesicIdentity(const MatrixXd& a, const MatrixXd& b, MatrixXd& g, const double alpha)
{
	(void)b;
	(void)alpha;
	g = MatrixXd::Identity(a.rows(), a.rows());
	return true;
}
//---------------------------------------------------------------------------------------------------
