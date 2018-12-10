///-------------------------------------------------------------------------------------------------
/// 
/// \file IMatrixClassifier.hpp
/// 
/// \brief Abstract class of Matrix Classifier
/// 
/// \author Thibaut Monseigne (Inria).
/// 
/// \version 1.0.
/// 
/// \date 10/12/2018.
/// 
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>
#include <vector>
#include "3rd-party/tinyxml2.h"

///----------------------------------------------------------------------------------------------------
///
/// <summary>	Abstract class of Matrix Classifier. </summary>
///
///----------------------------------------------------------------------------------------------------
class IMatrixClassifier
{
public:	
	/// <summary>	Number of class to classify. </summary>
	size_t m_ClassCount = 2;

	//***********************	
	//***** Constructor *****	
	//***********************	
	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Initializes a new instance of the <see cref="IMatrixClassifier"/> class. </summary>
	///
	///----------------------------------------------------------------------------------------------------
	IMatrixClassifier() = default;

	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Initializes a new instance of the <see cref="IMatrixClassifier"/> class and set the number of class. </summary>
	///
	/// <param name="classcount">	The number of class. </param>
	///
	///----------------------------------------------------------------------------------------------------
	explicit IMatrixClassifier(size_t classcount);
	
	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Finalizes an instance of the <see cref="IMatrixClassifier"/> class. </summary>
	///
	///----------------------------------------------------------------------------------------------------
	virtual ~IMatrixClassifier() = default;

	//**********************
	//***** Classifier *****
	//**********************
	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Sets the class count. </summary>
	///
	/// <param name="classcount">	The classcount. </param>
	///
	///----------------------------------------------------------------------------------------------------
	virtual void setClassCount(const size_t classcount);
	
	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Train the classifier with the dataset. </summary>
	///
	/// <param name="datasets">	The data set one class by row and trials on colums. </param>
	///
	/// <returns>	True if it succeeds, false if it fails. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	virtual bool train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets) = 0;
	
	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Classify the matrix and return the class id. </summary>
	///
	/// <param name="sample">	The sample to classify. </param>
	/// <param name="classid">	The predicted class. </param>
	///
	/// <returns>	True if it succeeds, false if it fails. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	virtual bool classify(const Eigen::MatrixXd& sample, size_t& classid) = 0;
	
	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Classify the matrix and return the class id, the distance and the probability of each class. </summary>
	///
	/// <param name="sample">		The sample to classify. </param>
	/// <param name="classid">		The predicted class. </param>
	/// <param name="distance">		The distance of the sample with each class. </param>
	/// <param name="probability">	The probability of the sample with each class. </param>
	///
	/// <returns>	True if it succeeds, false if it fails. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	virtual bool classify(const Eigen::MatrixXd& sample, size_t& classid, std::vector<double>& distance, std::vector<double>& probability) = 0;

	//***********************
	//***** XML Manager *****
	//***********************
	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Saves the classifier information in an XML file. </summary>
	///
	/// <param name="filename">	Filename. </param>
	///
	/// <returns>	True if it succeeds, false if it fails. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	virtual bool saveXML(const std::string& filename) = 0;
	
	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Loads the classifier information from an XML file. </summary>
	///
	/// <param name="filename">	Filename. </param>
	///
	/// <returns>	True if it succeeds, false if it fails. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	virtual bool loadXML(const std::string& filename) = 0;

	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Add the attribute on the first node (général informations as classifier type). </summary>
	///
	/// <param name="element">	Node to modify. </param>
	///
	/// <returns>	True if it succeeds, false if it fails. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	virtual bool saveHeaderAttribute(tinyxml2::XMLElement* element) const = 0;
	
	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Loads the attribute on the first node (général informations as classifier type). </summary>
	///
	/// <param name="element">	Node to read. </param>
	///
	/// <returns>	True if it succeeds, false if it fails. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	virtual bool loadHeaderAttribute(tinyxml2::XMLElement* element) = 0;
	
	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Saves the information of ith (\p index) class. </summary>
	///
	/// <param name="element">	Node of the class. </param>
	/// <param name="index">	Class number. </param>
	///
	/// <returns>	True if it succeeds, false if it fails. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	virtual bool saveClass(tinyxml2::XMLElement* element, const size_t index) const = 0;
	
	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Loads the information of ith (\p index) class. </summary>
	///
	/// <param name="element">	Node of the class. </param>
	/// <param name="index">	Class number. </param>
	///
	/// <returns>	True if it succeeds, false if it fails. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	virtual bool loadClass(tinyxml2::XMLElement* element, const size_t index) = 0;

	//*****************************
	//***** Override Operator *****
	//*****************************
	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Get the type of the classifier. </summary>
	///
	/// <returns>	The type in string. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	virtual std::string getType() const = 0;
	
	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Get the Classifier information for output. </summary>
	///
	/// <returns>	The Classifier print in stringstream. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	virtual std::stringstream print() const = 0;
};
