#include "Distance.hpp"
#include "Basics.hpp"
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;
using namespace std;

//---------------------------------------------------------------------------------------------------
double Distance(const MatrixXd& a, const MatrixXd& b, const EMetric metric)
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
double DistanceRiemann(const MatrixXd& a, const MatrixXd& b)
{
	const GeneralizedSelfAdjointEigenSolver<MatrixXd> es(a, b);
	const ArrayXd result = es.eigenvalues();
	return sqrt(result.log().square().sum());
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
double DistanceEuclidian(const MatrixXd& a, const MatrixXd& b) { return (b - a).norm(); }
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
double DistanceLogEuclidian(const MatrixXd& a, const MatrixXd& b) { return DistanceEuclidian(a.log(), b.log()); }
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
double DistanceLogDet(const MatrixXd& a, const MatrixXd& b) { return sqrt(log((0.5 * (a + b)).determinant()) - 0.5 * log(a.determinant() * b.determinant())); }
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
double DistanceKullback(const MatrixXd& a, const MatrixXd& b) { return 0.5 * ((b.inverse() * a).trace() - a.rows() + log(b.determinant() / a.determinant())); }
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
double DistanceKullbackSym(const MatrixXd& a, const MatrixXd& b) { return DistanceKullback(a, b) + DistanceKullback(b, a); }
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
double DistanceWasserstein(const MatrixXd& a, const MatrixXd& b)
{
	const MatrixXd sB = b.sqrt();
	return sqrt((a + b - 2 * (sB * a * sB).sqrt()).trace());
}
//---------------------------------------------------------------------------------------------------
