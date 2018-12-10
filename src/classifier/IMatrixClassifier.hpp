#pragma once

#include <Eigen/Dense>
#include <vector>
#include "3rd-party/tinyxml2.h"

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
	virtual bool classify(const Eigen::MatrixXd& sample, size_t& classid) = 0;
	virtual bool classify(const Eigen::MatrixXd& sample, size_t& classid, std::vector<double>& distance, std::vector<double>& probability) = 0;

	//***** XML *****
	virtual bool saveXML(const std::string& filename) = 0;
	virtual bool loadXML(const std::string& filename) = 0;

	virtual bool saveHeaderAttribute(tinyxml2::XMLElement* element) const = 0;
	virtual bool loadHeaderAttribute(tinyxml2::XMLElement* element) = 0;
	virtual bool saveClass(tinyxml2::XMLElement* element, const size_t index) const = 0;
	virtual bool loadClass(tinyxml2::XMLElement* element, const size_t index) = 0;

	//***** Override Operator *****
	virtual std::string getType() const = 0;
	virtual std::stringstream print() const = 0;
};
