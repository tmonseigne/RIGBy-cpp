#include "geometry/Basics.hpp"
#include <unsupported/Eigen/MatrixFunctions> // SQRT of Matrix

namespace Geometry {

//************************************************
//******************** Matrix ********************
//************************************************
//---------------------------------------------------------------------------------------------------
Eigen::MatrixXd AffineTransformation(const Eigen::MatrixXd& ref, const Eigen::MatrixXd& matrix)
{
	const Eigen::MatrixXd isR = ref.sqrt().inverse();	// Inverse Square root of Reference matrix => isR
	return isR * matrix * isR.transpose();				// Affine transformation : isR * sample * isR^T
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MatrixStandardization(Eigen::MatrixXd& matrix, const EStandardization standard)
{
	if (standard == EStandardization::Center) { return MatrixCenter(matrix); }
	if (standard == EStandardization::StandardScale) { return MatrixStandardization(matrix); }
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MatrixStandardization(const Eigen::MatrixXd& in, Eigen::MatrixXd& out, const EStandardization standard)
{
	out = in;
	return MatrixStandardization(out, standard);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MatrixCenter(Eigen::MatrixXd& matrix)
{
	for (size_t i = 0, r = matrix.rows(), c = matrix.cols(); i < r; ++i)
	{
		const double mu = matrix.row(i).mean();
		for (size_t j = 0; j < c; ++j) { matrix(i, j) -= mu; }
	}
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MatrixCenter(const Eigen::MatrixXd& in, Eigen::MatrixXd& out)
{
	out = in;
	return MatrixCenter(out);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MatrixStandardScaler(Eigen::MatrixXd& matrix)
{
	Eigen::RowVectorXd dummyScale;
	return MatrixStandardScaler(matrix, dummyScale);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MatrixStandardScaler(Eigen::MatrixXd& matrix, Eigen::RowVectorXd& scale)
{
	const size_t r = matrix.rows(), c = matrix.cols();
	std::vector<double> mu(r, 0), sigma(r, 0);
	scale.resize(r);

	for (size_t i = 0; i < r; ++i)
	{
		for (size_t j = 0; j < c; ++j)
		{
			const double value = matrix(i, j);
			mu[i] += value;
			sigma[i] += value * value;
		}

		mu[i] /= double(c);
		sigma[i] = sigma[i] / double(c) - mu[i] * mu[i];
		scale[i] = sigma[i] == 0 ? 1 : sqrt(sigma[i]);

		for (size_t j = 0; j < c; ++j) { matrix(i, j) = (matrix(i, j) - mu[i]) / scale[i]; }
	}
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MatrixStandardScaler(const Eigen::MatrixXd& in, Eigen::MatrixXd& out, Eigen::RowVectorXd& scale)
{
	out = in;
	return MatrixStandardScaler(out, scale);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MatrixStandardScaler(const Eigen::MatrixXd& in, Eigen::MatrixXd& out)
{
	Eigen::RowVectorXd dummyScale;
	return MatrixStandardScaler(in, out, dummyScale);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
std::string MatrixPrint(const Eigen::MatrixXd& matrix)
{
	std::stringstream sstream;
	sstream << matrix;
	return sstream.str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool AreEquals(const Eigen::MatrixXd& matrix1, const Eigen::MatrixXd& matrix2, const double precision)
{
	return matrix1.size() == matrix2.size() && (matrix1.size() == 0 || matrix1.isApprox(matrix2, precision));
}
//---------------------------------------------------------------------------------------------------

//************************************************
//************************************************
//************************************************

//*************************************************************
//******************** Index Manipulations ********************
//*************************************************************
//---------------------------------------------------------------------------------------------------
Eigen::RowVectorXd GetElements(const Eigen::RowVectorXd& row, const std::vector<size_t>& index)
{
	const size_t k = index.size();
	Eigen::RowVectorXd result(k);
	for (size_t i = 0; i < k; ++i) { result[i] = row[index[i]]; }
	return result;
}
//---------------------------------------------------------------------------------------------------
//*************************************************************
//*************************************************************
//*************************************************************

//***************************************************
//******************** Validates ********************
//***************************************************
//---------------------------------------------------------------------------------------------------
bool InRange(const double value, const double min, const double max) { return (min <= value && value <= max); }
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool AreNotEmpty(const std::vector<Eigen::MatrixXd>& matrices)
{
	if (matrices.empty()) { return false; }
	for (const auto& m : matrices) { if (!IsNotEmpty(m)) { return false; } }
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool IsNotEmpty(const Eigen::MatrixXd& matrix) { return (matrix.rows() != 0 && matrix.cols() != 0); }
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool HaveSameSize(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) { return (IsNotEmpty(a) && a.rows() == b.rows() && a.cols() == b.cols()); }
//---------------------------------------------------------------------------------------------------

bool IsSquare(const Eigen::MatrixXd& matrix) { return (IsNotEmpty(matrix) && matrix.rows() == matrix.cols()); }
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool AreSquare(const std::vector<Eigen::MatrixXd>& matrices)
{
	if (matrices.empty()) { return false; }
	for (const auto& m : matrices) { if (!IsSquare(m)) { return false; } }
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool HaveSameSize(const std::vector<Eigen::MatrixXd>& matrices)
{
	if (matrices.empty()) { return false; }
	const size_t r = matrices[0].rows(), c = matrices[0].cols();
	for (const auto& m : matrices) { if (size_t(m.rows()) != r || size_t(m.cols()) != c) { return false; } }
	return true;
}
//---------------------------------------------------------------------------------------------------
//***************************************************
//***************************************************
//***************************************************

//********************************************************
//******************** CSV MANAGEMENT ********************
//********************************************************
//---------------------------------------------------------------------------------------------------
std::vector<std::string> Split(const std::string& s, const std::string& sep)
{
	std::vector<std::string> result;
	std::string::size_type i       = 0, j;
	const std::string::size_type n = sep.size();

	while ((j = s.find(sep, i)) != std::string::npos)
	{
		result.emplace_back(s, i, j - i);			// Add part
		i = j + n;									// Update pos
	}
	result.emplace_back(s, i, s.size() - 1 - i);	// Last without \n
	return result;
}
//---------------------------------------------------------------------------------------------------
//********************************************************
//********************************************************
//********************************************************

}  // namespace Geometry
