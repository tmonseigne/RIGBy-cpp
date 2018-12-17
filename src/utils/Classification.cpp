#include "Classification.hpp"
#include "Covariance.hpp"
#include "Basics.hpp"
#include <iostream>

using namespace std;
using namespace Eigen;

bool LSQR(const std::vector<std::vector<Eigen::RowVectorXd>>& datasets, MatrixXd& weight)
{
	// Precomputation
	if (datasets.empty()) { return false; }
	const size_t nbClass = datasets.size(), nbFeatures = datasets[0][0].size();
	vector<size_t> nbSample(nbClass);
	size_t totalSample = 0;
	for (size_t k = 0; k < nbClass; ++k)
	{
		if (datasets[k].empty()) { return false; }
		nbSample[k] = datasets[k].size();
		totalSample += nbSample[k];
	}

	// Compute Class Euclidian mean
	MatrixXd mean = MatrixXd::Zero(nbClass, nbFeatures);
	for (size_t k = 0; k < nbClass; ++k)
	{
		for (size_t i = 0; i < nbSample[k]; ++i)
		{
			mean.row(k) += datasets[k][i];
		}
		mean.row(k) /= double(nbSample[k]);
	}

	// Compute Class Covariance
	MatrixXd cov = MatrixXd::Zero(nbFeatures, nbFeatures);
	for (size_t k = 0; k < nbClass; ++k)
	{
		//Fit Data to existing covariance matrix method
		MatrixXd classData(nbFeatures, nbSample[k]);
		for (size_t i = 0; i < nbSample[k]; ++i)
		{
			classData.col(i) = datasets[k][i];
		}

		// Standardize Features
		vector<double> scale;
		MatrixStandardScaler(classData, scale);

		//Compute Covariance of this class
		MatrixXd classCov;
		if (!CovarianceMatrix(classData, classCov, Estimator_LWF)) { return false; }

		// Rescale
		for (size_t i = 0; i < nbFeatures; ++i){
			for (size_t j = 0; j < nbFeatures; ++j){
				classCov(i, j) *= scale[i] * scale[j];
			}
		}

		//Add to cov with good weight
		cov += (double(nbSample[k]) / double(totalSample)) * classCov;
	}

	// linear least squares systems solver
	weight = cov.bdcSvd(ComputeThinU | ComputeThinV).solve(mean.transpose()).transpose();
	return true;
}
