#include "geometry/artifacts/CASR.hpp"

#include "geometry/Misc.hpp"
#include "geometry/Median.hpp"
#include "geometry/Covariance.hpp"
#include "geometry/Mean.hpp"
#include "geometry/classifier/IMatrixClassifier.hpp"

#include <boost/math/special_functions/detail/igamma_inverse.hpp>
#include <unsupported/Eigen/MatrixFunctions>

#include <cmath>
#include <numeric>

namespace Geometry {

///-------------------------------------------------------------------------------------------------
bool CASR::train(const std::vector<Eigen::MatrixXd>& dataset, const double rejectionLimit)
{
	if (dataset.empty() || dataset[0].size() == 0) { return false; }
	const size_t n = dataset.size();	// Number of samples
	m_nChannel     = dataset[0].rows();	// Number of channels

	//========== Compute the covariance matrix ==========
	std::vector<Eigen::MatrixXd> covs(n);
	//for (size_t i = 0; i < n; ++i) { if (!CovarianceMatrixLWF(dataset[i], covs[i])) { return false; } }	// We assume data is centered
	for (size_t i = 0; i < n; ++i) { if (!CovarianceMatrix(dataset[i], covs[i], EEstimator::LWF, EStandardization::Center)) { return false; } }

	//========== Compute Square Root of Median ==========
	if (!Median(covs, m_median)) { return false; }											// Geometric median independant of metric
	m_median = m_median.sqrt();

	//========== Compute Eigen vectors ==========
	Eigen::MatrixXd eigVector;
	std::vector<double> eigValues;
	sortedEigenVector(m_median, eigVector, eigValues, m_metric);							//Actually only Euclidian metric is implemented

	//========== Compute the ponderate dataset ==========
	std::vector<Eigen::MatrixXd> newDataset;
	newDataset.reserve(n);
	for (const auto& m : dataset) { newDataset.push_back((m.transpose() * eigVector)); }	// Multiply by eigen vector (we transpose to have channels in column
	for (auto& m : newDataset) { m = m.cwiseProduct(m); }									// Square new signal

	//========== Compute the "fit" distribution ==========
	// Compute the RMS of each channel for each sample
	std::vector<std::vector<double>> rms(m_nChannel, std::vector<double>(n));
	for (size_t i = 0; i < n; ++i) { for (size_t j = 0; j < m_nChannel; ++j) { rms[j][i] = sqrt(newDataset[i].col(j).mean()); } }

	// Compute the "fit" distribution
	std::vector<double> mu(m_nChannel, 0.0), sigma(m_nChannel, 0.0);
	for (size_t i = 0; i < m_nChannel; ++i) { FitDistribution(rms[i], mu[i], sigma[i]); }

	// Compute the threshold Matrix
	m_treshold = Eigen::MatrixXd::Zero(m_nChannel, m_nChannel);
	for (size_t i = 0; i < m_nChannel; ++i) { m_treshold(i, i) = mu[i] + rejectionLimit * sigma[i]; }
	m_treshold *= eigVector.transpose();

	// Initialize Reconstruction matrix and trivial
	m_r       = Eigen::MatrixXd::Identity(m_nChannel, m_nChannel);
	m_trivial = true;
	return true;
}

bool CASR::process(const Eigen::MatrixXd& in, Eigen::MatrixXd& out)
{
	// Check if input data is compatible with train data and if we don't limit so mutch the reconstruction
	out = in;
	if (size_t(out.rows()) != m_nChannel) { return false; }
	const size_t begin = size_t((1.0 - m_maxChannel) * double(m_nChannel));	// We define the number of channels to non reconstruct
	if (begin == m_nChannel) { return true; }
	if (m_r.size() == 0) { m_r = Eigen::MatrixXd::Identity(m_nChannel, m_nChannel); }

	// Compute Covariance matrix
	Eigen::MatrixXd cov;
	if (!CovarianceMatrix(in, cov, EEstimator::LWF, EStandardization::Center)) { return false; }
	if (m_cov.size() == 0) { m_cov = cov; }									// if first time
	else { if (!Mean({ m_cov, cov }, m_cov, m_metric)) { return false; } }	// else mean of the both

	// Compute Eigen vector & values
	Eigen::MatrixXd eigVector;
	std::vector<double> eigValues;
	sortedEigenVector(m_cov, eigVector, eigValues, m_metric);

	// Check if eigen values is over threshold computes during train (ponderate by eigen vector)
	Eigen::MatrixXd threshold = (m_treshold * eigVector).cwiseAbs2();
	bool trivial              = true;
	std::vector<bool> keep(m_nChannel, true);
	for (size_t i = begin; i < m_nChannel; ++i)
	{
		if (eigValues[i] >= threshold.col(i).sum())
		{
			keep[i] = false;
			trivial = false;
		}
	}

	// Check if All channels are clean
	if (trivial) { m_r = Eigen::MatrixXd::Identity(m_nChannel, m_nChannel); }
	else	// if not...
	{
		// Compute the reconstruction matrix with bad channels 
		Eigen::MatrixXd tmp = eigVector.transpose() * m_median;
		for (size_t i = begin; i < m_nChannel; ++i) { if (!keep[i]) { tmp.row(i).setZero(); } }
		const Eigen::MatrixXd newR = m_median * tmp.completeOrthogonalDecomposition().pseudoInverse() * eigVector.transpose();

		if (!m_trivial)
		{
			// Compute blend values for the samples
			const size_t nSample = in.cols();
			std::vector<double> blend(nSample);
			std::iota(blend.begin(), blend.end(), 1);	// Range 1 to nSample (inclued)
			for (auto& b : blend) { b = (1 - cos(M_PI * (b / double(nSample)))) / 2.0; }

			// Apply reconstruction ponderate by the blend (we considere the old reconstruction matrix for the second part)
			Eigen::MatrixXd t1 = newR * in;
			Eigen::MatrixXd t2 = m_r * in;
			for (size_t i = 0; i < nSample; ++i) { out.col(i) = (blend[i] * t1.col(i)) + ((1 - blend[i]) * t2.col(i)); }
		}
		m_r = newR;	// Update the reconstruction matrix
	}
	m_trivial = trivial;
	return true;
}

///-------------------------------------------------------------------------------------------------

bool CASR::setMatrices(const Eigen::MatrixXd& median, const Eigen::MatrixXd& threshold, const Eigen::MatrixXd& reconstruct,
					   const Eigen::MatrixXd& covariance)
{
	if (!IsSquare(median) || !HaveSameSize(median, threshold)
		|| (reconstruct.size() != 0 && !HaveSameSize(median, reconstruct))
		|| (covariance.size() != 0 && !HaveSameSize(median, covariance)))
	{
		std::cout << "All matrices must be square with same size (or empty for reconstruct and covariance matrix" << std::endl;
		return false;
	}
	m_nChannel = median.rows();
	m_median   = median;
	m_treshold = threshold;
	m_r        = reconstruct.size() != 0 ? reconstruct : Eigen::MatrixXd::Identity(m_nChannel, m_nChannel);
	m_cov      = covariance;
	m_trivial  = true;
	return true;
}
///-------------------------------------------------------------------------------------------------

//***********************
//***** XML Manager *****
//***********************

///-------------------------------------------------------------------------------------------------
bool CASR::saveXML(const std::string& filename) const
{
	tinyxml2::XMLDocument doc;
	// Create Root
	tinyxml2::XMLNode* root = doc.NewElement("ASR");			// Create root node
	doc.InsertFirstChild(root);									// Add root to XML

	tinyxml2::XMLElement* data = doc.NewElement("ASR-data");	// Create data node
	data->SetAttribute("metric", toString(m_metric).c_str());	// Set attribute metric
	data->SetAttribute("nChannel", int(m_nChannel));			// Set attribute nCHannel
	data->SetAttribute("maxChannel", int(m_maxChannel));		// Set attribute nCHannel
	data->SetAttribute("trivial", m_trivial);					// Set attribute nCHannel

	tinyxml2::XMLElement* median = doc.NewElement("Median");	// Create Median node
	if (!IMatrixClassifier::saveMatrix(median, m_median)) { return false; }	// Save Median Matrix
	data->InsertEndChild(median);								// Add Median node to data node

	tinyxml2::XMLElement* treshold = doc.NewElement("Treshold");	// Create Median node
	if (!IMatrixClassifier::saveMatrix(treshold, m_treshold)) { return false; }	// Save Median Matrix
	data->InsertEndChild(treshold);								// Add Median node to data node

	tinyxml2::XMLElement* r = doc.NewElement("R");				// Create Median node
	if (!IMatrixClassifier::saveMatrix(r, m_r)) { return false; }	// Save Median Matrix
	data->InsertEndChild(r);									// Add Median node to data node

	tinyxml2::XMLElement* cov = doc.NewElement("Cov");			// Create Median node
	if (!IMatrixClassifier::saveMatrix(cov, m_cov)) { return false; }	// Save Median Matrix
	data->InsertEndChild(cov);									// Add Median node to data node

	root->InsertEndChild(data);									// Add data to root
	return doc.SaveFile(filename.c_str()) == 0;					// save XML (if != 0 it means error)
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
bool CASR::loadXML(const std::string& filename)
{
	// Load File
	tinyxml2::XMLDocument xmlDoc;
	if (xmlDoc.LoadFile(filename.c_str()) != 0) { return false; }		// Check File Exist and Loading

	// Load Root
	tinyxml2::XMLNode* root = xmlDoc.FirstChild();						// Get Root Node
	if (root == nullptr) { return false; }								// Check Root Node Exist

	// Load Data
	tinyxml2::XMLElement* data = root->FirstChildElement("ASR-data");	// Get Data Node
	if (data == nullptr) { return false; }								// Check Root Node Exist
	m_metric     = StringToMetric(std::string(data->Attribute("metric")));
	m_nChannel   = data->IntAttribute("nChannel");
	m_maxChannel = data->IntAttribute("maxChannel");
	m_trivial    = data->BoolAttribute("trivial");

	tinyxml2::XMLElement* element = data->FirstChildElement("Median");	// Get Median Node
	if (element == nullptr) { return false; }							// Check if Node Exist
	if (!IMatrixClassifier::loadMatrix(element, m_median)) { return false; }	// Load Median Matrix

	element = data->FirstChildElement("Treshold");						// Get Treshold Node
	if (element == nullptr) { return false; }							// Check if Node Exist
	if (!IMatrixClassifier::loadMatrix(element, m_treshold)) { return false; }	// Load Treshold Matrix

	element = data->FirstChildElement("R");								// Get R Node
	if (element == nullptr) { return false; }							// Check if Node Exist
	if (!IMatrixClassifier::loadMatrix(element, m_r)) { return false; }		// Load R Matrix

	element = data->FirstChildElement("Cov");							// Get Cov Node
	if (element == nullptr) { return false; }							// Check if Node Exist
	if (!IMatrixClassifier::loadMatrix(element, m_cov)) { return false; }	// Load Cov Matrix

	return true;
}
///-------------------------------------------------------------------------------------------------

//*****************************
//***** Override Operator *****
//*****************************

///-------------------------------------------------------------------------------------------------
bool CASR::isEqual(const CASR& obj, const double precision) const
{
	return m_metric == obj.m_metric && m_nChannel == obj.m_nChannel
		   && abs(m_maxChannel - obj.m_maxChannel) < precision && m_trivial == obj.m_trivial
		   && AreEquals(m_median, obj.m_median, precision) && AreEquals(m_treshold, obj.m_treshold, precision)
		   && AreEquals(m_r, obj.m_r, precision) && AreEquals(m_cov, obj.m_cov, precision);
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
void CASR::copy(const CASR& obj)
{
	m_metric     = obj.m_metric;
	m_nChannel   = obj.m_nChannel;
	m_maxChannel = obj.m_maxChannel;
	m_trivial    = obj.m_trivial;
	m_median     = obj.m_median;
	m_treshold   = obj.m_treshold;
	m_r          = obj.m_r;
	m_cov        = obj.m_cov;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
std::stringstream CASR::print() const
{
	std::stringstream ss;
	ss << "Metric : " << toString(m_metric) << std::endl;
	if (m_nChannel == 0) { ss << "Train not done" << std::endl; }
	else
	{
		ss << "Train done." << std::endl;
		ss << size_t(m_maxChannel * double(m_nChannel)) << "/" << m_nChannel << " channels can be reconstruted." << std::endl;
		ss << "Median matrix is : " << std::endl << m_median << std::endl;
		ss << "Treshold matrix is : " << std::endl << m_treshold << std::endl;
		if (m_cov.size() == 0) { ss << "No process launched yet." << std::endl; }
		else
		{
			ss << "Last sample " << (m_trivial ? "was" : "wasn't") << " trivial." << std::endl;
			ss << "Last Reconstruction Matrix : " << std::endl << m_r << std::endl;
			ss << "Last Covariance Matrix : " << std::endl << m_cov << std::endl;
		}
	}
	return ss;
}
///-------------------------------------------------------------------------------------------------

}  // namespace Geometry
