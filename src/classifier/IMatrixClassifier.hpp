///-------------------------------------------------------------------------------------------------
/// 
/// \file IMatrixClassifier.hpp
/// \brief Abstract class of Matrix Classifier
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 10/12/2018.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>
#include <vector>
#include "3rd-party/tinyxml2.h"
#include "utils/Metrics.hpp"

/// <summary>	Abstract class of Matrix Classifier. </summary>
class IMatrixClassifier
{
public:	
	/// <summary>	Number of class to classify. </summary>
	size_t m_ClassCount = 2;

	/// <summary>	Metric to use to calculate means and distances (see also <see cref="EMetrics" />). </summary>
	EMetrics m_Metric = Metric_Riemann;

	//***********************	
	//***** Constructor *****	
	//***********************	
	/// <summary>	Default constructor. Initializes a new instance of the <see cref="IMatrixClassifier"/> class. </summary>
	IMatrixClassifier() = default;

	/// <summary>	Default Copy constructor. Initializes a new instance of the <see cref="IMatrixClassifier"/> class. </summary>
	/// <param name="obj">	Initial object. </param>
	IMatrixClassifier(const IMatrixClassifier& obj);

	/// <summary>	Initializes a new instance of the <see cref="IMatrixClassifier"/> class and set members. </summary>
	/// <param name="classcount">	The number of class. </param>
	/// <param name="metric">	Metric to use to calculate means (see also <see cref="EMetrics" />). </param>
	explicit IMatrixClassifier(size_t classcount, EMetrics metric);

	/// <summary>	Finalizes an instance of the <see cref="IMatrixClassifier"/> class. </summary>
	virtual ~IMatrixClassifier() = default;

	//**********************
	//***** Classifier *****
	//**********************
	/// <summary>	Sets the class count. </summary>
	/// <param name="classcount">	The classcount. </param>
	virtual void setClassCount(size_t classcount);
	
	/// <summary>	Train the classifier with the dataset. \n
	/// Each row is the trials of each class. The trainer compute the mean of each row with the <see cref="EMetrics" /> in <see cref="m_Metric"/> member.
	/// </summary>
	/// <param name="datasets">	The data set one class by row and trials on colums. </param>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	virtual bool train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets) = 0;
	
	/// <summary>	Classify the matrix and return the class id (override of same function with all argument). </summary>
	/// <param name="sample">	The sample to classify. </param>
	/// <param name="classid">	The predicted class. </param>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	virtual bool classify(const Eigen::MatrixXd& sample, size_t& classid) = 0;
	
	/// <summary>	Classify the matrix and return the class id, the distance and the probability of each class. </summary>
	/// <param name="sample">		The sample to classify. </param>
	/// <param name="classid">		The predicted class. </param>
	/// <param name="distance">		The distance of the sample with each class. </param>
	/// <param name="probability">	The probability of the sample with each class. </param>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	virtual bool classify(const Eigen::MatrixXd& sample, size_t& classid, std::vector<double>& distance, std::vector<double>& probability) = 0;

	//***********************
	//***** XML Manager *****
	//***********************
	/// <summary>	Saves the classifier information in an XML file. </summary>
	/// <param name="filename">	Filename. </param>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	virtual bool saveXML(const std::string& filename) = 0;
	
	/// <summary>	Loads the classifier information from an XML file. </summary>
	/// <param name="filename">	Filename. </param>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	virtual bool loadXML(const std::string& filename) = 0;

	/// <summary>	Add the attribute on the first node (general informations as classifier type). </summary>
	/// <param name="element">	Node to modify. </param>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	virtual bool saveHeaderAttribute(tinyxml2::XMLElement* element) const = 0;
	
	/// <summary>	Loads the attribute on the first node (general informations as classifier type). </summary>
	/// <param name="element">	Node to read. </param>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	virtual bool loadHeaderAttribute(tinyxml2::XMLElement* element) = 0;
	
	/// <summary>	Format the Matrix for XML Saving. </summary>
	/// <param name="in">	Matrix. </param>
	/// <param name="out">	Stringstream. </param>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	static bool convertMatrixToXMLFormat(const Eigen::MatrixXd& in, std::stringstream& out);

	/// <summary>	Fill the Matrix From XML Format. </summary>
	/// <param name="in">	Stringstream. </param>
	/// <param name="out">	Matrix. </param>
	/// <param name="rows">	Number of rows. </param>
	/// <param name="cols">	Number of cols. </param>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	static bool convertXMLFormatToMatrix(std::stringstream& in, Eigen::MatrixXd& out, size_t rows, size_t cols);

	/// <summary>	Saves matrix. </summary>
	/// <param name="element">	Matrix Node. </param>
	/// <param name="matrix">	Matrix to save. </param>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	static bool saveMatrix(tinyxml2::XMLElement* element, const Eigen::MatrixXd& matrix);

	/// <summary>	Load matrix. </summary>
	/// <param name="element">	Matrix Node. </param>
	/// <param name="matrix">	Matrix to load. </param>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	static bool loadMatrix(tinyxml2::XMLElement* element, Eigen::MatrixXd& matrix);


	//*****************************
	//***** Override Operator *****
	//*****************************
	/// <summary>	Check if object are equals (with a precision tolerance). </summary>
	/// <param name="obj">			The second object. </param>
	/// <param name="precision">	Precision for matrix comparison. </param>
	/// <returns>	True if the two elements are equals (with a precision tolerance). </returns>
	bool isEqual(const IMatrixClassifier& obj, const double precision = 1e-6) const;

	/// <summary>	Copy object value. </summary>
	/// <param name="obj">	The object to copy. </param>
	void copy(const IMatrixClassifier& obj);

	/// <summary>	Get the type of the classifier. </summary>
	/// <returns>	The type in string. </returns>
	virtual std::string getType() const = 0;
	
	/// <summary>	Get the Classifier information for output. </summary>
	/// <returns>	The Classifier print in stringstream. </returns>
	virtual std::stringstream print() const = 0;

	/// <summary>	Override the affectation operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	The copied object. </returns>
	IMatrixClassifier& operator=(const IMatrixClassifier& obj)
	{
		copy(obj);
		return *this;
	}

	/// <summary>	Override the egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	True if the two <see cref="CMatrixClassifierMDM"/> are equals. </returns>
	bool operator==(const IMatrixClassifier& obj) const { return isEqual(obj); }

	/// <summary>	Override the not egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	True if the two <see cref="CMatrixClassifierMDM"/> are diffrents. </returns>
	bool operator!=(const IMatrixClassifier& obj) const { return !isEqual(obj); }

	/// <summary>	Override the ostream operator. </summary>
	/// <param name="os">	The ostream. </param>
	/// <param name="obj">	The object. </param>
	/// <returns>	Return the modified ostream. </returns>
	friend std::ostream& operator <<(std::ostream& os, const IMatrixClassifier& obj)
	{
		os << obj.print().str();
		return os;
	}
};
