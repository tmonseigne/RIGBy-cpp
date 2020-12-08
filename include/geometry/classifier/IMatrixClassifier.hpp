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
#include "geometry/Metrics.hpp"
#include "geometry/3rd-party/tinyxml2.h"

namespace Geometry {

///-------------------------------------------------------------------------------------------------
/// <summary>	Enumeration of Adaptation Methods for classifier. </summary>
enum class EAdaptations
{
	None,			///< No Adaptation.
	Supervised,		///< Supervised Adaptation.
	Unsupervised	///< Unsupervised Adaptation.
};

/// <summary>	Convert adaptations to string. </summary>
/// <param name="type">	The type of adaptation. </param>
/// <returns>	<c>std::string</c> </returns>
inline std::string toString(const EAdaptations type)
{
	switch (type)
	{
		case EAdaptations::None: return "No";
		case EAdaptations::Supervised: return "Supervised";
		case EAdaptations::Unsupervised: return "Unsupervised";
	}
	return "Invalid";
}

/// <summary>	Convert string to adaptations. </summary>
/// <param name="type">	The type of adaptation. </param>
/// <returns>	<see cref="EAdaptations"/> </returns>
inline EAdaptations StringToAdaptation(const std::string& type)
{
	if (type == "No") { return EAdaptations::None; }
	if (type == "Supervised") { return EAdaptations::Supervised; }
	return EAdaptations::Unsupervised;
}
///-------------------------------------------------------------------------------------------------

///-------------------------------------------------------------------------------------------------
/// <summary>	Enumeration of Matrix Classifiers. </summary>
enum class EMatrixClassifiers
{
	MDM,				///< Minimum Distance To Mean (MDM) Classifier.
	MDM_Rebias,			///< Minimum Distance To Mean Rebias (MDM Rebias) Classifier.
	FgMDM_RT,			///< Minimum Distance to Mean with geodesic filtering (FgMDM) (Real Time adaptation assumed).
	FgMDM,				///< Minimum Distance to Mean with geodesic filtering (FgMDM).
	FgMDM_RT_Rebias,	///< Minimum Distance to Mean with geodesic filtering & Rebias adaptation (FgMDM Rebias) (Real Time adaptation assumed).
	FgMDM_Rebias		///< Minimum Distance to Mean with geodesic filtering & Rebias adaptation (FgMDM Rebias).
};


/// <summary>	Convert Matrix Classifiers to string. </summary>
/// <param name="type">	The type of classifier. </param>
/// <returns>	<c>std::string</c> </returns>
inline std::string toString(const EMatrixClassifiers type)
{
	switch (type)
	{
		case EMatrixClassifiers::MDM: return "Minimum Distance to Mean (MDM)";
		case EMatrixClassifiers::MDM_Rebias: return "Minimum Distance to Mean Rebias (MDM Rebias)";
		case EMatrixClassifiers::FgMDM_RT: return "Minimum Distance to Mean with geodesic filtering (FgMDM) (Real Time adaptation assumed)";
		case EMatrixClassifiers::FgMDM: return "Minimum Distance to Mean with geodesic filtering (FgMDM)";
		case EMatrixClassifiers::FgMDM_RT_Rebias: return
					"Minimum Distance to Mean with geodesic filtering Rebias (FgMDM Rebias) (Real Time adaptation assumed)";
		case EMatrixClassifiers::FgMDM_Rebias: return "Minimum Distance to Mean with geodesic filtering Rebias (FgMDM Rebias)";
	}
	return "Invalid";
}

/// <summary>	Convert string to Matrix Classifiers. </summary>
/// <param name="type">	The type of classifier. </param>
/// <returns>	<see cref="EMatrixClassifiers"/> </returns>
inline EMatrixClassifiers StringToMatrixClassifier(const std::string& type)
{
	if (type == "Minimum Distance to Mean (MDM)") { return EMatrixClassifiers::MDM; }
	if (type == "Minimum Distance to Mean Rebias (MDM Rebias)") { return EMatrixClassifiers::MDM_Rebias; }
	if (type == "Minimum Distance to Mean with geodesic filtering (FgMDM) (Real Time adaptation assumed)") { return EMatrixClassifiers::FgMDM_RT; }
	if (type == "Minimum Distance to Mean with geodesic filtering (FgMDM)") { return EMatrixClassifiers::FgMDM; }
	if (type == "Minimum Distance to Mean with geodesic filtering Rebias (FgMDM Rebias) (Real Time adaptation assumed)")
	{
		return EMatrixClassifiers::FgMDM_RT_Rebias;
	}
	return EMatrixClassifiers::FgMDM_Rebias;
}
///-------------------------------------------------------------------------------------------------

/// <summary>	Format the Eigen matrix with Full Precision. </summary>
#define MATRIX_FORMAT Eigen::IOFormat(-2, 0, " ", "\n", "", "", "", "")

///-------------------------------------------------------------------------------------------------
/// <summary>	Abstract class of Matrix Classifier. </summary>
class IMatrixClassifier
{
public:
	//****************************	
	//***** Static Functions *****	
	//****************************
	/// <summary>	Format the Matrix for XML Saving. </summary>
	/// <param name="in">	Matrix. </param>
	/// <param name="out">	Stringstream. </param>
	/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
	static bool convertMatrixToXMLFormat(const Eigen::MatrixXd& in, std::stringstream& out);

	/// <summary>	Fill the Matrix From XML Format. </summary>
	/// <param name="in">	Stringstream. </param>
	/// <param name="out">	Matrix. </param>
	/// <param name="rows">	Number of rows. </param>
	/// <param name="cols">	Number of cols. </param>
	/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
	static bool convertXMLFormatToMatrix(std::stringstream& in, Eigen::MatrixXd& out, size_t rows, size_t cols);

	/// <summary>	Saves matrix. </summary>
	/// <param name="element">	Matrix Node. </param>
	/// <param name="matrix">	Matrix to save. </param>
	/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
	static bool saveMatrix(tinyxml2::XMLElement* element, const Eigen::MatrixXd& matrix);

	/// <summary>	Load matrix. </summary>
	/// <param name="element">	Matrix Node. </param>
	/// <param name="matrix">	Matrix to load. </param>
	/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
	static bool loadMatrix(tinyxml2::XMLElement* element, Eigen::MatrixXd& matrix);

	//***********************	
	//***** Constructor *****	
	//***********************	
	/// <summary>	Default constructor. Initializes a new instance of the <see cref="IMatrixClassifier"/> class. </summary>
	IMatrixClassifier() = default;

	/// <summary>	Default Copy constructor. Initializes a new instance of the <see cref="IMatrixClassifier"/> class. </summary>
	/// <param name="obj">	Initial object. </param>
	IMatrixClassifier(const IMatrixClassifier& obj) { *this = obj; }

	/// <summary>	Initializes a new instance of the <see cref="IMatrixClassifier"/> class and set members. </summary>
	/// <param name="nbClass">	The number of classes. </param>
	/// <param name="metric">	Metric to use to calculate means (see also <see cref="EMetric" />). </param>
	explicit IMatrixClassifier(size_t nbClass, EMetric metric);

	/// <summary>	Finalizes an instance of the <see cref="IMatrixClassifier"/> class. </summary>
	virtual ~IMatrixClassifier() = default;

	//**********************
	//***** Classifier *****
	//**********************
	virtual size_t getClassCount() const { return m_nbClass; }	///< Get the class count.
	virtual void setClassCount(size_t nbClass);					///< Set the class count.

	/// <summary>	Train the classifier with the dataset. </summary>
	/// <param name="datasets">	The dataset one class by row and trials on colums. </param>
	/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
	virtual bool train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets) = 0;

	/// <summary>	Classify the matrix and return the class id (override of same function with all argument). </summary>
	/// <param name="sample">		The sample to classify. </param>
	/// <param name="classId">		The predicted class. </param>
	/// <param name="adaptation">	Adaptation method for the classfier <see cref="EAdaptations" />. </param>
	/// <param name="realClassId">	The expected class id if supervised adaptation. </param>
	/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
	/// <seealso cref="classify(const Eigen::MatrixXd&, size_t&, std::vector<double>&, std::vector<double>&, EAdaptations, const size_t&)"/>
	virtual bool classify(const Eigen::MatrixXd& sample, size_t& classId,
						  EAdaptations adaptation = EAdaptations::None, const size_t& realClassId = std::numeric_limits<size_t>::max());

	/// <summary>	Classify the matrix and return the class id, the distance and the probability of each class. </summary>
	/// <param name="sample">		The sample to classify. </param>
	/// <param name="classId">		The predicted class. </param>
	/// <param name="distance">		The distance of the sample with each class. </param>
	/// <param name="probability">	The probability of the sample with each class. </param>
	/// <param name="adaptation">	Adaptation method for the classfier <see cref="EAdaptations" />. </param>
	/// <param name="realClassId">	The expected class id if supervised adaptation. </param>
	/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>	
	virtual bool classify(const Eigen::MatrixXd& sample, size_t& classId, std::vector<double>& distance, std::vector<double>& probability,
						  EAdaptations adaptation = EAdaptations::None, const size_t& realClassId = std::numeric_limits<size_t>::max()) = 0;

	//***********************
	//***** XML Manager *****
	//***********************
	/// <summary>	Saves the classifier information in an XML file. </summary>
	/// <param name="filename">	Filename. </param>
	/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
	virtual bool saveXML(const std::string& filename) const;

	/// <summary>	Loads the classifier information from an XML file. </summary>
	/// <param name="filename">	Filename. </param>
	/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
	virtual bool loadXML(const std::string& filename);


	//*****************************
	//***** Override Operator *****
	//*****************************

	/// <summary>	Check if object are equals (with a precision tolerance). </summary>
	/// <param name="obj">			The second object. </param>
	/// <param name="precision">	Precision for matrix comparison. </param>
	/// <returns>	<c>True</c> if the two elements are equals (with a precision tolerance). </returns>
	bool isEqual(const IMatrixClassifier& obj, double precision = 1e-6) const;

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

	/// <summary>	Override the equal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	<c>True</c> if the two <see cref="IMatrixClassifier"/> are equals. </returns>
	bool operator==(const IMatrixClassifier& obj) const { return isEqual(obj); }

	/// <summary>	Override the not equal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	<c>True</c> if the two objects are diffrents. </returns>
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

protected:
	/// <summary>	Prints the header informations. </summary>
	/// <returns>	Header informations in stringstream. </returns>
	virtual std::stringstream printHeader() const;

	/// <summary>	Prints the Additional informations. </summary>
	/// <returns>	Additional informations in stringstream. </returns>
	virtual std::stringstream printAdditional() const { return std::stringstream(); }

	/// <summary>	Prints the Classes informations. </summary>
	/// <returns>	Classes informations in stringstream. </returns>
	virtual std::stringstream printClasses() const { return std::stringstream(); }

	//***********************
	//***** XML Manager *****
	//***********************
	/// <summary>	Add the attribute on the first node (general informations as classifier type, number of class...).
	///
	/// -# The type of the classifier : <see cref="getType"/>
	/// -# The number of classes : <see cref="m_nbClass"/>
	/// -# The metric to use : <see cref="m_metric"/>
	/// </summary>
	/// <param name="data">	Node to modify. </param>
	/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
	virtual bool saveHeader(tinyxml2::XMLElement* data) const;

	/// <summary>	Loads the attribute on the first node (general informations as classifier type, number of class...).
	///
	/// -# Check the type : <see cref="getType"/>
	/// -# The number of classes : <see cref="m_nbClass"/>
	/// -# The metric to use : <see cref="m_metric"/>
	/// </summary>
	/// <param name="data">	Node to read. </param>
	/// <returns>	<c>True</c> if it succeeds, <c>False</c> otherwise. </returns>
	virtual bool loadHeader(tinyxml2::XMLElement* data);

	/// <summary>	Save Additionnal informations (none at this level). </summary>
	/// <returns>	<c>True</c>. </returns>
	virtual bool saveAdditional(tinyxml2::XMLDocument& /*doc*/, tinyxml2::XMLElement* /*data*/) const { return true; }

	/// <summary>	Load Additionnal informations (none at this level). </summary>
	/// <returns>	<c>True</c>. </returns>
	virtual bool loadAdditional(tinyxml2::XMLElement* /*data*/) { return true; }

	/// <summary>	Save Classes informations (none at this level). </summary>
	/// <returns>	<c>True</c>. </returns>
	virtual bool saveClasses(tinyxml2::XMLDocument& /*doc*/, tinyxml2::XMLElement* /*data*/) const { return true; }

	/// <summary>	Load Classes informations (none at this level). </summary>
	/// <returns>	<c>True</c>. </returns>
	virtual bool loadClasses(tinyxml2::XMLElement* /*data*/) { return true; }


	//*********************	
	//***** Variables *****
	//*********************	
	size_t m_nbClass = 2;					///< Number of classes to classify. 
	EMetric m_metric = EMetric::Riemann;	///< Metric to use to calculate means and distances (see also <see cref="EMetric" />).
};

}  // namespace Geometry
