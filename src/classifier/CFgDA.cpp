#include "CFgDA.hpp"
#include "utils/Featurization.hpp"
#include "utils/Mean.hpp"

using namespace std;
using namespace Eigen;

bool CFgDA::compute(const vector<MatrixXd>& dataset)
{
	if (!Mean(dataset, m_Ref, m_Metric)) { return false; }	// Compute Reference matrix
	const size_t nbSample = dataset.size();
	vector<RowVectorXd> ts(nbSample);
	for (size_t i = 0; i < nbSample; ++i)					// Transform to the Tangent Space
	{
		if (!TangentSpace(dataset[i], ts[i], m_Ref)) { return false; }
	}
	if (!lsqr(ts)) { return false; }						// Compute Weight with lsqr

	// Transforme weight	self._W = numpy.dot(numpy.dot(W.T, numpy.linalg.pinv(numpy.dot(W, W.T))), W)
	const RowVectorXd w = m_Weight, wT = m_Weight.transpose();
	m_Weight = (wT * /*Compute the (Moore-Penrose) pseudo-inverse of a matrix.*/(w * wT)) * w;
	return true;
}
///----------------------------------------------------------------------------------------------------

bool CFgDA::lsqr(const std::vector<RowVectorXd>& dataset)
{
	(void)dataset;
	/*
	 *        self.means_ = _class_means(X, y)
        self.covariance_ = _class_cov(X, y, self.priors_, shrinkage)
        self.coef_ = linalg.lstsq(self.covariance_, self.means_.T)[0].T
        self.intercept_ = (-0.5 * np.diag(np.dot(self.means_, self.coef_.T)) +
                           np.log(self.priors_))
	 */
	// Classical Mean

	/*
	const size_t k = dataset.size(),							// Number of Vector			=> K
				 n = dataset[0].size();							// Number of Features		=> N
	RowVectorXd mean = RowVectorXd::Zero(n);					// Initial Mean
	for (const auto& data : dataset) { mean += data; }			// Sum of Vector
	mean /= double(k);											// Normalization
	*/
	return true;
}
///----------------------------------------------------------------------------------------------------

bool CFgDA::filter(const MatrixXd& in, MatrixXd& out) const
{
	RowVectorXd ts;
	if (!TangentSpace(in, ts, m_Ref)) { return false; }		// Transform input in tangent space
	ts = ts.cwiseProduct(m_Weight);							// Apply weight
	return UnTangentSpace(ts, out, m_Ref);					// Transform in covariance matrix
}
///----------------------------------------------------------------------------------------------------

bool CFgDA::operator==(const CFgDA& obj) const
{
	(void)obj;
	return true;
}
///----------------------------------------------------------------------------------------------------

bool CFgDA::operator!=(const CFgDA& obj) const
{
	(void)obj;
	return true;
}
///----------------------------------------------------------------------------------------------------

std::stringstream CFgDA::print() const
{
	stringstream ss;
	ss << "Metric : " << MetricToString(m_Metric) << endl;
	ss << "Weight : " << m_Weight << endl;
	return ss;
}
///----------------------------------------------------------------------------------------------------

std::ostream& operator<<(std::ostream& os, const CFgDA& obj)
{
	os << obj.print().str();
	return os;
}
///----------------------------------------------------------------------------------------------------
