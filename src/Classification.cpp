#include "geometry/Classification.hpp"
#include "geometry/Covariance.hpp"
#include "geometry/Basics.hpp"

namespace Geometry {

///-------------------------------------------------------------------------------------------------
bool LSQR(const std::vector<std::vector<Eigen::RowVectorXd>>& datasets, Eigen::MatrixXd& weight)
{
	// Precomputation
	if (datasets.empty()) { return false; }
	const size_t nbClass = datasets.size(), nbFeatures = datasets[0][0].size();
	std::vector<size_t> nbSample(nbClass);
	size_t totalSample = 0;
	for (size_t k = 0; k < nbClass; ++k)
	{
		if (datasets[k].empty()) { return false; }
		nbSample[k] = datasets[k].size();
		totalSample += nbSample[k];
	}

	// Compute Class Euclidian mean
	Eigen::MatrixXd mean = Eigen::MatrixXd::Zero(nbClass, nbFeatures);
	for (size_t k = 0; k < nbClass; ++k)
	{
		for (size_t i = 0; i < nbSample[k]; ++i) { mean.row(k) += datasets[k][i]; }
		mean.row(k) /= double(nbSample[k]);
	}

	// Compute Class Covariance
	Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(nbFeatures, nbFeatures);
	for (size_t k = 0; k < nbClass; ++k)
	{
		//Fit Data to existing covariance matrix method
		Eigen::MatrixXd classData(nbFeatures, nbSample[k]);
		for (size_t i = 0; i < nbSample[k]; ++i) { classData.col(i) = datasets[k][i]; }

		// Standardize Features
		Eigen::RowVectorXd scale;
		MatrixStandardScaler(classData, scale);

		//Compute Covariance of this class
		Eigen::MatrixXd classCov;
		if (!CovarianceMatrix(classData, classCov, EEstimator::LWF)) { return false; }

		// Rescale
		for (size_t i = 0; i < nbFeatures; ++i) { for (size_t j = 0; j < nbFeatures; ++j) { classCov(i, j) *= scale[i] * scale[j]; } }

		//Add to cov with good weight
		cov += (double(nbSample[k]) / double(totalSample)) * classCov;
	}

	// linear least squares systems solver
	// Chosen solver with the performance table of this page : https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
	weight = cov.colPivHouseholderQr().solve(mean.transpose()).transpose();
	//weight = cov.completeOrthogonalDecomposition().solve(mean.transpose()).transpose();
	//weight = cov.bdcSvd(ComputeThinU | ComputeThinV).solve(mean.transpose()).transpose();

	// Treat binary case as a special case
	if (nbClass == 2)
	{
		const Eigen::MatrixXd tmp = weight.row(1) - weight.row(0);	// Need to use a tmp variable otherwise sometimes error
		weight                    = tmp;
	}
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool FgDACompute(const std::vector<std::vector<Eigen::RowVectorXd>>& datasets, Eigen::MatrixXd& weight)
{
	// Compute LSQR Weight
	Eigen::MatrixXd w;
	if (!LSQR(datasets, w)) { return false; }
	const size_t nbClass = w.rows();

	// Transform to FgDA Weight
	const Eigen::MatrixXd wT = w.transpose();
	weight                   = (wT * (w * wT).colPivHouseholderQr().solve(Eigen::MatrixXd::Identity(nbClass, nbClass))) * w;
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool FgDAApply(const Eigen::RowVectorXd& in, Eigen::RowVectorXd& out, const Eigen::MatrixXd& weight)
{
	if (in.cols() != weight.rows()) { return false; }
	out = in * weight;
	return true;
}
///-------------------------------------------------------------------------------------------------

}  // namespace Geometry
