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
#include <limits>
#include "3rd-party/tinyxml2.h"
#include "utils/Metrics.hpp"

/// <summary>	Enumeration of Adaptation Methods for classifier. </summary>
enum EAdaptations
{
	/// <summary>	No Adaptation. </summary>
	Adaptation_None,
	/// <summary>	Supervised Adaptation. </summary>
	Adaptation_Supervised,
	/// <summary>	Unsupervised Adaptation. </summary>
	Adaptation_Unsupervised
};

/// <summary>	Abstract class of Matrix Classifier. </summary>
class IMatrixClassifier
{
public:
	//***********************	
	//***** Constructor *****	
	//***********************	
	/// <summary>	Default constructor. Initializes a new instance of the <see cref="IMatrixClassifier"/> class. </summary>
	IMatrixClassifier() = default;

	/// <summary>	Default Copy constructor. Initializes a new instance of the <see cref="IMatrixClassifier"/> class. </summary>
	/// <param name="obj">	Initial object. </param>
	IMatrixClassifier(const IMatrixClassifier& obj) { *this = obj; }

	/// <summary>	Don't override move constructor. </summary>
	/// <param name="obj">	Initial object. </param>
	IMatrixClassifier(IMatrixClassifier&& obj) = default;

	/// <summary>	Initializes a new instance of the <see cref="IMatrixClassifier"/> class and set members. </summary>
	/// <param name="nbClass">	The number of classes. </param>
	/// <param name="metric">	Metric to use to calculate means (see also <see cref="EMetrics" />). </param>
	explicit IMatrixClassifier(const size_t nbClass, const EMetrics metric);

	/// <summary>	Finalizes an instance of the <see cref="IMatrixClassifier"/> class. </summary>
	virtual ~IMatrixClassifier() = default;

	//**********************
	//***** Classifier *****
	//**********************
	/// <summary>	Sets the class count. </summary>
	/// <param name="nbClass">	The number of Classes. </param>
	virtual void setClassCount(const size_t nbClass);

	/// <summary>	get the class count. </summary>
	virtual size_t getClassCount() const { return m_nbClass; }
	
	/// <summary>	Train the classifier with the dataset. </summary>
	/// <param name="datasets">	The data set one class by row and trials on colums. </param>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	virtual bool train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets) = 0;

	/// <summary>	Classify the matrix and return the class id (override of same function with all argument). </summary>
	/// <param name="sample">		The sample to classify. </param>
	/// <param name="classId">		The predicted class. </param>
	/// <param name="adaptation">	Adaptation method for the classfier <see cref="EAdaptations" />. </param>
	/// <param name="realClassId">	The expected class id if supervised adaptation. </param>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	/// <seealso cref="classify(const Eigen::MatrixXd&, size_t&, std::vector<double>&, std::vector<double>&, EAdaptations, const size_t&)"/>
	virtual bool classify(const Eigen::MatrixXd& sample, size_t& classId,
						  const EAdaptations adaptation = Adaptation_None, const size_t& realClassId = std::numeric_limits<std::size_t>::max());

	/// <summary>	Classify the matrix and return the class id, the distance and the probability of each class. </summary>
	/// <param name="sample">		The sample to classify. </param>
	/// <param name="classId">		The predicted class. </param>
	/// <param name="distance">		The distance of the sample with each class. </param>
	/// <param name="probability">	The probability of the sample with each class. </param>
	/// <param name="adaptation">	Adaptation method for the classfier <see cref="EAdaptations" />. </param>
	/// <param name="realClassId">	The expected class id if supervised adaptation. </param>
	/// <returns>	True if it succeeds, false if it fails. </returns>	
	virtual bool classify(const Eigen::MatrixXd& sample, size_t& classId, std::vector<double>& distance, std::vector<double>& probability,
						  const EAdaptations adaptation = Adaptation_None, const size_t& realClassId = std::numeric_limits<std::size_t>::max()) = 0;	

	//***********************
	//***** XML Manager *****
	//***********************
	/// <summary>	Saves the classifier information in an XML file. </summary>
	/// <param name="filename">	Filename. </param>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	virtual bool saveXML(const std::string& filename);
	
	/// <summary>	Loads the classifier information from an XML file. </summary>
	/// <param name="filename">	Filename. </param>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	virtual bool loadXML(const std::string& filename);


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
	virtual std::stringstream print() const;

	/// <summary>	Override the affectation operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	The copied object. </returns>
	IMatrixClassifier& operator=(const IMatrixClassifier& obj)
	{
		copy(obj);		
		return *this;
	}

	/// <summary>	Don't Override the move operator. </summary>
	IMatrixClassifier& operator=(IMatrixClassifier&& obj) = default;

	/// <summary>	Override the egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	True if the two <see cref="IMatrixClassifier"/> are equals. </returns>
	bool operator==(const IMatrixClassifier& obj) const { return isEqual(obj); }

	/// <summary>	Override the not egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	True if the two objects are diffrents. </returns>
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

	//*********************	
	//***** Variables *****
	//*********************	
	/// <summary>	Metric to use to calculate means and distances (see also <see cref="EMetrics" />). </summary>
	EMetrics m_Metric = Metric_Riemann;

protected:	
	/// <summary>	Prints the header informations.</summary>
	/// <returns>	Header informations in stringstream</returns>
	virtual std::stringstream printHeader() const;

	/// <summary>	Prints the Additional informations.</summary>
	/// <returns>	Additional informations in stringstream</returns>
	virtual std::stringstream printAdditional() const { return std::stringstream(); }

	/// <summary>	Prints the Classes informations.</summary>
	/// <returns>	Classes informations in stringstream</returns>
	virtual std::stringstream printClasses() const { return std::stringstream(); }

	//***********************
	//***** XML Manager *****
	//***********************
	/// <summary>	Add the attribute on the first node (general informations as classifier type, number of class...).
	///
	/// -# The type of the classifier : <see cref="getType"/>
	/// -# The number of classes : <see cref="m_nbClass"/>
	/// -# The metric to use : <see cref="m_Metric"/>
	/// </summary>
	/// <param name="data">	Node to modify. </param>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	virtual bool saveHeader(tinyxml2::XMLElement* data) const;

	/// <summary>	Loads the attribute on the first node (general informations as classifier type, number of class...).
	///
	/// -# Check the type : <see cref="getType"/>
	/// -# The number of classes : <see cref="m_nbClass"/>
	/// -# The metric to use : <see cref="m_Metric"/>
	/// </summary>
	/// <param name="data">	Node to read. </param>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	virtual bool loadHeader(tinyxml2::XMLElement* data);

	/// <summary>	Save Additionnal informations (none at this level). </summary>
	/// <returns>	True. </returns>
	virtual bool saveAdditional(tinyxml2::XMLDocument& /*doc*/, tinyxml2::XMLElement* /*data*/) const { return true; }
	
	/// <summary>	Load Additionnal informations (none at this level). </summary>
	/// <returns>	True. </returns>
	virtual bool loadAdditional(tinyxml2::XMLElement* /*data*/) { return true; }
	
	/// <summary>	Save Classes informations (none at this level). </summary>
	/// <returns>	True. </returns>
	virtual bool saveClasses(tinyxml2::XMLDocument& /*doc*/, tinyxml2::XMLElement* /*data*/) const { return true; }
	
	/// <summary>	Load Classes informations (none at this level). </summary>
	/// <returns>	True. </returns>
	virtual bool loadClasses(tinyxml2::XMLElement* /*data*/) { return true; }

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

	//*********************	
	//***** Variables *****
	//*********************	
	/// <summary>	Number of classes to classify. </summary>
	size_t m_nbClass = 2;
};
