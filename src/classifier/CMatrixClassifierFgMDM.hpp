///-------------------------------------------------------------------------------------------------
/// 
/// \file CMatrixClassifierMDM.hpp
/// 
/// \brief Class of Minimum Distance to Mean with geodesic filtering (FgMDM) Classifier
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

#include "CMatrixClassifierMDM.hpp"

///-------------------------------------------------------------------------------------------------
/// 
/// <summary>	Class of Minimum Distance to Mean with geodesic filtering (FgMDM) Classifier. </summary>
/// 
/// <seealso cref="IMatrixClassifier" />
/// 
///-------------------------------------------------------------------------------------------------
class CMatrixClassifierFgMDM : public CMatrixClassifierMDM
{
public:
	Eigen::MatrixXd m_Ref,
					m_Weight;

	//***********************	
	//***** Constructor *****
	//***********************	
	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Initializes a new instance of the <see cref="CMatrixClassifierFgMDM"/> class. </summary>
	///
	///----------------------------------------------------------------------------------------------------
	CMatrixClassifierFgMDM() { }

	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Initializes a new instance of the <see cref="CMatrixClassifierFgMDM"/> class and set the number of class and the metric. </summary>
	///
	/// <param name="classcount">	The number of class. </param>
	/// <param name="metric">	Metric to use to calculate means (see also <see cref="EMetrics" />). </param>
	///
	///----------------------------------------------------------------------------------------------------
	explicit CMatrixClassifierFgMDM(const size_t classcount, const EMetrics metric) : CMatrixClassifierMDM(classcount, metric) { }

	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Finalizes an instance of the <see cref="CMatrixClassifierFgMDM"/> class. </summary>
	///
	///----------------------------------------------------------------------------------------------------
	~CMatrixClassifierFgMDM() override = default;

	//********************
	//***** Computes *****
	//********************
	bool computeFgDA(const std::vector<std::vector<Eigen::MatrixXd>>& datasets);

	//********************
	//********************
	//********************

	//**********************
	//***** Classifier *****
	//**********************
	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Sets the class count. </summary>
	///
	/// <param name="classcount">	The classcount. </param>
	///
	/// \remark resize the <see cref="m_Means"/> vector of Matrix.
	///
	///----------------------------------------------------------------------------------------------------
	void setClassCount(const size_t classcount) override { CMatrixClassifierMDM::setClassCount(classcount); }

	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Train the classifier with the dataset. 
	///
	/// Each row is the trials of each class. The trainer compute the mean of each row with the <see cref="EMetrics" /> in <see cref="m_Metric"/> member.
	/// </summary>
	///
	/// <param name="datasets">	The data set one class by row and trials on colums. </param>
	///
	/// <returns>	True if it succeeds, false if it fails. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	bool train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets) override;

	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Classify the matrix and return the class id (override of same function with all argument). </summary>
	///
	/// <param name="sample">	The sample to classify. </param>
	/// <param name="classid">	The predicted class. </param>
	///
	/// <returns>	True if it succeeds, false if it fails. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	bool classify(const Eigen::MatrixXd& sample, size_t& classid) override;

	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Classify the matrix and return the class id, the distance and the probability of each class. 
	///
	/// Compute the distance between the sample and each mean matrix.\n
	/// The class with the closest mean is the predicted class.\n
	/// The distance is returned.\n
	/// The probability \f$ \mathcal{P}_i \f$ to be the class \f$ i \f$ is compute as :
	/// \f[
	/// p_i = \frac{d_{\text{min}}}{d_i}\\
	/// \mathcal{P}_i =  \frac{p_i}{\sum{\left(p_i\right)}}
	/// \f]\n
	/// <b>Remark</b> : The probability is normalized \f$ \sum{\left(\mathcal{P}_i\right)} = 1 \f$
	///	</summary>
	///
	/// <param name="sample">		The sample to classify. </param>
	/// <param name="classid">		The predicted class. </param>
	/// <param name="distance">		The distance of the sample with each class. </param>
	/// <param name="probability">	The probability of the sample with each class. </param>
	///
	/// <returns>	True if it succeeds, false if it fails. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	bool classify(const Eigen::MatrixXd& sample, size_t& classid, std::vector<double>& distance, std::vector<double>& probability) override;

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
	bool saveXML(const std::string& filename) override;

	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Loads the classifier information from an XML file. </summary>
	///
	/// <param name="filename">	Filename. </param>
	///
	/// <returns>	True if it succeeds, false if it fails. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	bool loadXML(const std::string& filename) override;

	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Add the attribute on the first node.
	///
	/// -# The type of the classifier : FgMDM
	/// -# The number of class : <see cref="m_ClassCount"/>
	/// -# The metric to use : <see cref="m_Metric"/>
	/// </summary>
	///
	/// <param name="element">	Node to modify. </param>
	///
	/// <returns>	True if it succeeds, false if it fails. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	bool saveHeaderAttribute(tinyxml2::XMLElement* element) const override;

	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Loads the attribute on the first node.
	///
	/// -# Check the type : FgMDM
	/// -# The number of class : <see cref="m_ClassCount"/>
	/// -# The metric to use : <see cref="m_Metric"/>
	/// </summary>
	///
	/// <param name="element">	Node to read. </param>
	///
	/// <returns>	True if it succeeds, false if it fails. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	bool loadHeaderAttribute(tinyxml2::XMLElement* element) override;

	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Saves the information of ith (\p index) class. 
	///
	/// -# Save the id of the class \f$[0-\f$<see cref="m_ClassCount"/>\f$]\f$.
	/// -# Save the size \f$N\f$ of the matrix (Squared Matrix of \f$N \times N\f$ double).
	/// -# The matrix with space separator (and new line for each row for readability).
	/// </summary>
	///
	/// <param name="element">	Node of the class. </param>
	/// <param name="index">	Class number. </param>
	///
	/// <returns>	True if it succeeds, false if it fails. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	bool saveClass(tinyxml2::XMLElement* element, const size_t index) const override;

	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Loads the information of ith (\p index) class. 
	///
	/// -# Check the index.
	/// -# Load the matrix.
	/// </summary>
	///
	/// <param name="element">	Node of the class. </param>
	/// <param name="index">	Class number. </param>
	///
	/// <returns>	True if it succeeds, false if it fails. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	bool loadClass(tinyxml2::XMLElement* element, const size_t index) override;


	//*****************************
	//***** Override Operator *****
	//*****************************
	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Override the egal operator (with 1e-6 precision). </summary>
	///
	/// <param name="obj">	The second object. </param>
	///
	/// <returns>	True if the two <see cref="CMatrixClassifierFgMDM"/> are equals. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	bool operator==(const CMatrixClassifierFgMDM& obj) const;

	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Override the not egal operator (with 1e-6 precision). </summary>
	///
	/// <param name="obj">	The second object. </param>
	///
	/// <returns>	True if the two <see cref="CMatrixClassifierFgMDM"/> are diffrents. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	bool operator!=(const CMatrixClassifierFgMDM& obj) const;

	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Get the type of the classifier. </summary>
	///
	/// <returns>	Minimum Distance to Mean with geodesic filtering. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	std::string getType() const override { return "Minimum Distance to Mean with geodesic filtering"; }
	
	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Get the Classifier information for output. </summary>
	///
	/// <returns>	The Classifier print in stringstream. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	std::stringstream print() const override;

	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Override the ostream operator. </summary>
	///
	/// <param name="os">	The ostream. </param>
	/// <param name="obj">	The object. </param>
	///
	/// <returns>	Return the modified ostream. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	friend std::ostream& operator <<(std::ostream& os, const CMatrixClassifierFgMDM& obj);
};
