#include "Classification.hpp"
#include "Covariance.hpp"
#include "Basics.hpp"
#include <iostream>

using namespace std;
using namespace Eigen;

bool LSQR(const std::vector<std::vector<RowVectorXd>>& datasets, MatrixXd& weight)
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

	cout << "nbClass : " << nbClass << endl
		<< "nbFeatures : " << nbFeatures << endl
		<< "totalSample : " << totalSample << endl;


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

	cout << "Mean : " << endl << mean << endl;


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
		RowVectorXd scale;
		cout << "classData : " << k << endl << classData << endl;
		MatrixStandardScaler(classData, scale);
		cout << "classData standard : " << k << endl << classData << endl;
		cout << "classData scale : " << k << endl << MatrixXd(scale) << endl;
		cout << "classData scale t : " << k << endl << MatrixXd(scale.transpose()) << endl;

		//Compute Covariance of this class
		MatrixXd classCov;
		if (!CovarianceMatrix(classData, classCov, Estimator_LWF)) { return false; }
		cout << "classCov : " << k << endl << classCov << endl;

		// Rescale
		//classCov = MatrixXd(scale) * classCov * MatrixXd(scale.transpose());
		for (size_t i = 0; i < nbFeatures; ++i)
		{
			for (size_t j = 0; j < nbFeatures; ++j)
			{
				classCov(i, j) *= scale[i] * scale[j];
			}
		}
		cout << "classCov : " << k << endl << classCov << endl;

		//Add to cov with good weight
		cov += (double(nbSample[k]) / double(totalSample)) * classCov;
	}

	cout << "cov : " << endl << cov << endl;

	// linear least squares systems solver
	weight = cov.bdcSvd(ComputeThinU | ComputeThinV).solve(mean.transpose()).transpose();
	cout << "Weights" << endl << weight << endl;
	return true;
}
