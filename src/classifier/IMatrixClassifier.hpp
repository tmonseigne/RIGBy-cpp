#pragma once

#include <Eigen/Dense>
#include <vector>

class IMatrixClassifier
{
public:
	size_t m_ClassCount = 2;

	//***** Constructor *****
	IMatrixClassifier() = default;
	explicit IMatrixClassifier(size_t classcount);
	virtual ~IMatrixClassifier() = default;

	//***** Classifier *****
	virtual void setClassCount(const size_t classcount);
	virtual bool train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets) = 0;
	virtual bool classify(const Eigen::MatrixXd& sample, uint32_t& classid) = 0;
	virtual bool classify(const Eigen::MatrixXd& sample, uint32_t& classid, std::vector<double>& distance, std::vector<double>& probability) = 0;

	//***** XML *****
	virtual bool saveXML(const std::string& filename) = 0;
	virtual bool loadXML(const std::string& filename) = 0;
};
