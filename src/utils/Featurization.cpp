#include "Featurization.hpp"
#include <unsupported/Eigen/MatrixFunctions>
#include "Basics.hpp"

using namespace Eigen;
using namespace std;

#ifndef M_SQRT2
#define M_SQRT2 1.4142135623730950488016887242097
#endif

//---------------------------------------------------------------------------------------------------
bool Featurization(const MatrixXd& matrix, RowVectorXd& rowVector, const bool tangent, const MatrixXd& ref)
{
	if (tangent) { return TangentSpace(matrix, rowVector, ref); }
	return SqueezeUpperTriangle(matrix, rowVector, true);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool UnFeaturization(const RowVectorXd& rowVector, MatrixXd& matrix, const bool tangent, const MatrixXd& ref)
{
	if (tangent) { return UnTangentSpace(rowVector, matrix, ref); }
	return UnSqueezeUpperTriangle(rowVector, matrix, true);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool SqueezeUpperTriangle(const MatrixXd& matrix, RowVectorXd& rowVector, const bool rowMajor)
{
	if (!isSquare(matrix)) { return false; }					// Verification
	const size_t n = matrix.rows();								// Number of Features			=> N
	rowVector.resize(n * (n + 1) / 2);							// Resize

	size_t idx = 0;												// Row Index					=> idx
	if (rowMajor)												// Row Major Method
	{
		for (size_t i = 0; i < n; ++i)
		{
			for (size_t j = i; j < n; ++j) { rowVector[idx++] = matrix(i, j); }
		}
	}
	else														// Diagonal Method
	{
		for (size_t i = 0; i < n; ++i)
		{
			for (size_t j = i; j < n; ++j) { rowVector[idx++] = matrix(j, j - i); }
		}
	}
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool UnSqueezeUpperTriangle(const RowVectorXd& rowVector, MatrixXd& matrix, const bool rowMajor)
{
	const size_t nR = rowVector.size(),							// Size of Row					=> Nr
				 n = int((sqrt(1 + 8 * nR) - 1) / 2);			// Number of Features			=> N
	if (n == 0) { return false; }								// Verification
	matrix.setZero(n, n);										// Init

	size_t idx = 0;												// Row Index					=> idx
	if (rowMajor)												// Row Major Method
	{
		for (size_t i = 0; i < n; ++i)
		{
			for (size_t j = i; j < n; ++j) { matrix(j, i) = matrix(i, j) = rowVector[idx++]; }
		}
	}
	else														// Diagonal Method
	{
		for (size_t i = 0; i < n; ++i)
		{
			for (size_t j = i; j < n; ++j) { matrix(j - i, j) = matrix(j, j - i) = rowVector[idx++]; }
		}
	}
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool TangentSpace(const MatrixXd& matrix, RowVectorXd& rowVector, const MatrixXd& ref)
{
	if (!isSquare(matrix)) { return false; }					// Verification
	const size_t n = matrix.rows();								// Number of Features			=> N

	const MatrixXd sC = (ref.size() == 0) ? MatrixXd::Identity(n, n) : MatrixXd(ref.sqrt()),
				   isC = sC.inverse(),							// Inverse Square root of ref	=> isC
				   mJ = (isC * matrix * isC).log(),				// Transformation Matrix		=> mJ
				   mCoeffs = M_SQRT2 * MatrixXd(MatrixXd::Ones(n, n).triangularView<StrictlyUpper>()) + MatrixXd::Identity(n, n);

	RowVectorXd vJ, vCoeffs;
	if (!SqueezeUpperTriangle(mJ, vJ, true)) { return false; }	// Get upper triangle of J		=> vJ
	if (!SqueezeUpperTriangle(mCoeffs, vCoeffs, true)) { return false; }	// ... of Coefs		=> vCoeffs
	rowVector = vCoeffs.cwiseProduct(vJ);						// element-wise multiplication
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool UnTangentSpace(const RowVectorXd& rowVector, MatrixXd& matrix, const MatrixXd& ref)
{
	const size_t n = matrix.rows();								// Number of Features			=> N
	if (!UnSqueezeUpperTriangle(rowVector, matrix)) { return false; }

	const MatrixXd sC = (ref.size() == 0) ? MatrixXd::Identity(n, n) : MatrixXd(ref.sqrt()),
				   coeffs = MatrixXd(matrix.triangularView<StrictlyUpper>()) / M_SQRT2;

	matrix = sC * (MatrixXd(matrix.diagonal().asDiagonal()) + coeffs + coeffs.transpose()).exp() * sC;
	return true;
}
//---------------------------------------------------------------------------------------------------
