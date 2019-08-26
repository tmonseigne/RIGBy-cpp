#include "Basics.hpp"
#include <unsupported/Eigen/MatrixFunctions> // SQRT of Matrix

using namespace Eigen;
using namespace std;


//************************************************
//******************** Matrix ********************
//************************************************
//---------------------------------------------------------------------------------------------------
MatrixXd AffineTransformation(const MatrixXd& ref, const MatrixXd& matrix)
{
	const MatrixXd isR = ref.sqrt().inverse();	// Inverse Square root of Reference matrix => isR
	return isR * matrix * isR.transpose();		// Affine transformation : isR * sample * isR^T
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MatrixStandardization(MatrixXd& matrix, const EStandardization standard)
{
	if (standard == Standardization_Center) { return MatrixCenter(matrix); }
	if (standard == Standardization_StandardScale) { return MatrixStandardization(matrix); }
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MatrixStandardization(const MatrixXd& in, MatrixXd& out, const EStandardization standard)
{
	out = in;
	return MatrixStandardization(out, standard);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MatrixCenter(MatrixXd& matrix)
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
bool MatrixCenter(const MatrixXd& in, MatrixXd& out)
{
	out = in;
	return MatrixCenter(out);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MatrixStandardScaler(MatrixXd& matrix)
{
	RowVectorXd dummyScale;
	return MatrixStandardScaler(matrix, dummyScale);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MatrixStandardScaler(MatrixXd& matrix, RowVectorXd& scale)
{
	const size_t r = matrix.rows(), c = matrix.cols();
	vector<double> mu(r, 0), sigma(r, 0);
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
bool MatrixStandardScaler(const MatrixXd& in, MatrixXd& out, RowVectorXd& scale)
{
	out = in;
	return MatrixStandardScaler(out, scale);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool MatrixStandardScaler(const MatrixXd& in, MatrixXd& out)
{
	RowVectorXd dummyScale;
	return MatrixStandardScaler(in, out, dummyScale);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
string MatrixPrint(const MatrixXd& matrix)
{
	std::stringstream sstream;
	sstream << matrix;
	return sstream.str();
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool AreEquals(const MatrixXd& matrix1, const MatrixXd& matrix2, const double precision)
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
RowVectorXd GetElements(const RowVectorXd& row, const std::vector<size_t>& index)
{
	const size_t k = index.size();
	RowVectorXd result(k);
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
bool inRange(const double value, const double min, const double max)
{
	return (min <= value && value <= max);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool areNotEmpty(const vector<MatrixXd>& matrices)
{
	if (matrices.empty()) { return false; }
	for (const auto& m : matrices) { if (!isNotEmpty(m)) { return false; } }
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool isNotEmpty(const MatrixXd& matrix)
{
	return (matrix.rows() != 0 && matrix.cols() != 0);
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool haveSameSize(const MatrixXd& a, const MatrixXd& b)
{
	return (isNotEmpty(a) && a.rows() == b.rows() && a.cols() == b.cols());
}
//---------------------------------------------------------------------------------------------------

bool isSquare(const MatrixXd& matrix)
{
	return (isNotEmpty(matrix) && matrix.rows() == matrix.cols());
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool areSquare(const vector<MatrixXd>& matrices)
{
	if (matrices.empty()) { return false; }
	for (const auto& m : matrices) { if (!isSquare(m)) { return false; } }
	return true;
}
//---------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------
bool haveSameSize(const std::vector<MatrixXd>& matrices)
{
	if (matrices.empty()) { return false; }
	const size_t R = matrices[0].rows(), C = matrices[0].cols();
	for (const auto& m : matrices) { if (size_t(m.rows()) != R || size_t(m.cols()) != C) { return false; } }
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
vector<string> Split(const string& s, const string& sep)
{
	vector<string> result;
	string::size_type i       = 0, j;
	const string::size_type n = sep.size();

	while ((j = s.find(sep, i)) != string::npos)
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
