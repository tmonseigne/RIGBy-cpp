#include "Classification.hpp"
#include "Covariance.hpp"
#include "Mean.hpp"
#include "Basics.hpp"
#include <iostream>

using namespace std;
using namespace Eigen;

///-------------------------------------------------------------------------------------------------
bool LSQR(const std::vector<std::vector<RowVectorXd>>& datasets, MatrixXd& weight)
{
	// Compute Class Euclidian mean
	MatrixXd mean;
	if (!MeanClass(datasets, mean)) { return false; }

	// Compute Class Covariance
	MatrixXd cov;
	if (!CovarianceClass(datasets, cov)) { return false; }

	// linear least squares systems solver
	// Chosen solver with the performance table of this page : https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
	weight = cov.colPivHouseholderQr().solve(mean.transpose()).transpose();
	//weight = cov.completeOrthogonalDecomposition().solve(mean.transpose()).transpose();
	//weight = cov.bdcSvd(ComputeThinU | ComputeThinV).solve(mean.transpose()).transpose();

	// Treat binary case as a special case
	if (datasets.size() == 2)	// if two class
	{
		const MatrixXd tmp = weight.row(1) - weight.row(0);	// Need to use a tmp variable otherwise sometimes error
		weight             = tmp;
	}
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool EigenSolv(const std::vector<std::vector<RowVectorXd>>& datasets, MatrixXd& weight)
{
	/*
	 *  def _solve_eigen(self, X, y, shrinkage):
        """Eigenvalue solver.
        The eigenvalue solver computes the optimal solution of the Rayleigh
        coefficient (basically the ratio of between class scatter to within
        class scatter). This solver supports both classification and
        dimensionality reduction (with optional shrinkage).
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values.
        shrinkage : string or float, optional
            Shrinkage parameter, possible values:
              - None: no shrinkage (default).
              - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
              - float between 0 and 1: fixed shrinkage constant.
        Notes
        -----
        This solver is based on [1]_, section 3.8.3, pp. 121-124.
        References
        ----------
        .. [1] R. O. Duda, P. E. Hart, D. G. Stork. Pattern Classification
           (Second Edition). John Wiley & Sons, Inc., New York, 2001. ISBN
           0-471-05669-3.
        """
        self.means_ = _class_means(X, y)
        self.covariance_ = _class_cov(X, y, self.priors_, shrinkage)

        Sw = self.covariance_  # within scatter
        St = _cov(X, shrinkage)  # total scatter
        Sb = St - Sw  # between scatter

        evals, evecs = linalg.eigh(Sb, Sw)
        self.explained_variance_ratio_ = np.sort(evals / np.sum(evals)
                                                 )[::-1][:self._max_components]
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors

        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = (-0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(self.priors_))

	 */
	// Compute Class Euclidian mean
	MatrixXd mean;
	if (!MeanClass(datasets, mean)) { return false; }
	// Compute Class Covariance
	MatrixXd sw;
	if (!CovarianceClass(datasets, sw)) { return false; }

	// Compute global covariance matrix
	const std::vector<RowVectorXd> all = Vector2DTo1D(datasets);
	const size_t nbData                = all.size(), nbFeatures = all[0].size();
	MatrixXd classData(nbFeatures, nbData);
	for (size_t i = 0; i < nbData; ++i) { classData.col(i) = all[i]; }
	
	// Compute difference
	MatrixXd st;
	if (!CovarianceMatrix(classData, st, Estimator_LWF)) { return false; }
	const MatrixXd sb = st - sw;
	
	//eigen values
	std::vector<std::tuple<double, VectorXd>> eigens;
	SelfAdjointEigenSolver<MatrixXd> solver(sb);
	if (solver.info() != Success) { return false; }

	VectorXd values  = solver.eigenvalues();
	MatrixXd vectors = solver.eigenvectors();
	eigens.reserve(values.size());
	for (int i = 0; i < values.size(); i++) { eigens.push_back(make_tuple(values[i], vectors.row(i))); }

	const auto eigenSort = [&](const std::tuple<double, VectorXd>& a, const std::tuple<double, VectorXd>& b) { return std::get<0>(a) > std::get<0>(b); };
	std::sort(eigens.begin(), eigens.end(), eigenSort);
	for (size_t i = 0; i < eigens.size(); ++i) { vectors.row(i) = std::get<1>(eigens[i]); }
	weight = (mean * vectors) * vectors.transpose();

	cout << "------------------------ MeanClass ------------------------" << endl;
	cout << mean << endl;
	cout << "------------------------ CovarianceClass ------------------------" << endl;
	cout << sw << endl;
	cout << "------------------------ st ------------------------" << endl;
	cout << st << endl;
	cout << "------------------------ sb ------------------------" << endl;
	cout << sb << endl;
	cout << "------------------------ eigens ------------------------" << endl;
	for (const auto& e : eigens) { cout << std::get<0>(e) << "\t" << std::get<1>(e) << endl; }
	cout << "------------------------ weight ------------------------" << endl;
	cout << weight << endl;
	
	// Treat binary case as a special case
	if (datasets.size() == 2)	// if two class
	{
		const MatrixXd tmp = weight.row(1) - weight.row(0);	// Need to use a tmp variable otherwise sometimes error
		weight = tmp;
	}
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool FgDACompute(const vector<vector<RowVectorXd>>& datasets, MatrixXd& weight, const ESolver solver)
{
	// Compute LSQR Weight
	MatrixXd w;
	switch (solver)
	{
		case Solver_Eigen:
			if (!EigenSolv(datasets, w)) { return false; }
			break;
		case Solver_LSQR:
		default:
			if (!LSQR(datasets, w)) { return false; }
	}
	if (!LSQR(datasets, w)) { return false; }
	const size_t nbClass = w.rows();

	// Transform to FgDA Weight
	const MatrixXd wT = w.transpose();
	weight            = (wT * (w * wT).colPivHouseholderQr().solve(MatrixXd::Identity(nbClass, nbClass))) * w;
	return true;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool FgDAApply(const RowVectorXd& in, RowVectorXd& out, const MatrixXd& weight)
{
	if (in.cols() != weight.rows()) { return false; }
	out = in * weight;
	return true;
}
///-------------------------------------------------------------------------------------------------
