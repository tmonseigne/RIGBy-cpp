#include "geometry/Featurization.hpp"
#include <unsupported/Eigen/MatrixFunctions>
#include "geometry/Basics.hpp"

namespace Geometry {

#ifndef M_SQRT2
#define M_SQRT2 1.4142135623730950488016887242097
#endif

//---------------------------------------------------------------------------------------------------
bool Featurization(const Eigen::MatrixXd& in, Eigen::RowVectorXd& out, const bool tangent, const Eigen::MatrixXd& ref)
{
	if (tangent) { return TangentSpace(in, out, ref); }
	return SqueezeUpperTriangle(in, out, true);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool UnFeaturization(const Eigen::RowVectorXd& in, Eigen::MatrixXd& out, const bool tangent, const Eigen::MatrixXd& ref)
{
	if (tangent) { return UnTangentSpace(in, out, ref); }
	return UnSqueezeUpperTriangle(in, out, true);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool SqueezeUpperTriangle(const Eigen::MatrixXd& in, Eigen::RowVectorXd& out, const bool rowMajor)
{
	if (!IsSquare(in)) { return false; }						// Verification
	const size_t n = in.rows();									// Number of Features			=> N
	out.resize(n * (n + 1) / 2);								// Resize

	size_t idx = 0;												// Row Index					=> idx
	// Row Major or Diagonal Method
	if (rowMajor) { for (size_t i = 0; i < n; ++i) { for (size_t j = i; j < n; ++j) { out[idx++] = in(i, j); } } }
	else { for (size_t i = 0; i < n; ++i) { for (size_t j = i; j < n; ++j) { out[idx++] = in(j, j - i); } } }
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool UnSqueezeUpperTriangle(const Eigen::RowVectorXd& in, Eigen::MatrixXd& out, const bool rowMajor)
{
	const size_t nR = in.size(),								// Size of Row					=> Nr
				 n  = int((sqrt(1 + 8 * nR) - 1) / 2);			// Number of Features			=> N
	if (n == 0) { return false; }								// Verification
	out.setZero(n, n);											// Init

	size_t idx = 0;												// Row Index					=> idx
	// Row Major or Diagonal Method
	if (rowMajor) { for (size_t i = 0; i < n; ++i) { for (size_t j = i; j < n; ++j) { out(j, i) = out(i, j) = in[idx++]; } } }
	else { for (size_t i = 0; i < n; ++i) { for (size_t j = i; j < n; ++j) { out(j - i, j) = out(j, j - i) = in[idx++]; } } }
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool TangentSpace(const Eigen::MatrixXd& in, Eigen::RowVectorXd& out, const Eigen::MatrixXd& ref)
{
	if (!IsSquare(in)) { return false; }						// Verification
	const size_t n = in.rows();									// Number of Features			=> N

	const Eigen::MatrixXd sC      = (ref.size() == 0) ? Eigen::MatrixXd::Identity(n, n) : Eigen::MatrixXd(ref.sqrt()),
						  isC     = sC.inverse(),				// Inverse Square root of ref	=> isC
						  mJ      = (isC * in * isC).log(),		// Transformation Matrix		=> mJ
						  mCoeffs = M_SQRT2 * Eigen::MatrixXd(Eigen::MatrixXd::Ones(n, n).triangularView<Eigen::StrictlyUpper>())
									+ Eigen::MatrixXd::Identity(n, n);

	Eigen::RowVectorXd vJ, vCoeffs;
	if (!SqueezeUpperTriangle(mJ, vJ, true)) { return false; }	// Get upper triangle of J		=> vJ
	if (!SqueezeUpperTriangle(mCoeffs, vCoeffs, true)) { return false; }	// ... of Coefs		=> vCoeffs
	out = vCoeffs.cwiseProduct(vJ);								// element-wise multiplication
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool UnTangentSpace(const Eigen::RowVectorXd& in, Eigen::MatrixXd& out, const Eigen::MatrixXd& ref)
{
	const size_t n = out.rows();								// Number of Features			=> N
	if (!UnSqueezeUpperTriangle(in, out)) { return false; }

	const Eigen::MatrixXd sC     = (ref.size() == 0) ? Eigen::MatrixXd::Identity(n, n) : Eigen::MatrixXd(ref.sqrt()),
						  coeffs = Eigen::MatrixXd(out.triangularView<Eigen::StrictlyUpper>()) / M_SQRT2;

	out = sC * (Eigen::MatrixXd(out.diagonal().asDiagonal()) + coeffs + coeffs.transpose()).exp() * sC;
	return true;
}
//---------------------------------------------------------------------------------------------------

}  // namespace Geometry
