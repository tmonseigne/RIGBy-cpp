///-------------------------------------------------------------------------------------------------
/// 
/// \file CMatrixClassifierMDM.hpp
/// \brief Class of Minimum Distance to Mean (MDM) Classifier
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 10/12/2018.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include "IMatrixClassifier.hpp"
#include "utils/Metrics.hpp"

/// <summary>	Class of Minimum Distance to Mean (MDM) Classifier. </summary>
/// <seealso cref="IMatrixClassifier" />
class CMatrixClassifierMDM : public IMatrixClassifier
{
public:
	/// <summary>	Mean Matrix of each class. </summary>
	std::vector<Eigen::MatrixXd> m_Means;

	//***********************	
	//***** Constructor *****
	//***********************	
	/// <summary>	Default constructor. Initializes a new instance of the <see cref="CMatrixClassifierMDM"/> class. </summary>
	CMatrixClassifierMDM();

	/// <summary>	Default Copy constructor. Initializes a new instance of the <see cref="CMatrixClassifierMDM"/> class. </summary>
	/// \copydetails IMatrixClassifier(const IMatrixClassifier&)
	CMatrixClassifierMDM(const CMatrixClassifierMDM& obj);

	/// <summary>	Initializes a new instance of the <see cref="CMatrixClassifierMDM"/> class and set base members. </summary>
	/// \copydetails IMatrixClassifier(const size_t, const EMetrics)
	explicit CMatrixClassifierMDM(const size_t classcount, const EMetrics metric);

	/// <summary>	Finalizes an instance of the <see cref="CMatrixClassifierMDM"/> class. </summary>
	/// <remarks>	clear the <see cref="m_Means"/> vector of Matrix. </remarks>
	~CMatrixClassifierMDM() override;

	//**********************
	//***** Classifier *****
	//**********************
	/// <summary>	Sets the class count. </summary>
	/// \copydetails IMatrixClassifier::setClassCount(const size_t)
	/// <remarks>	resize the <see cref="m_Means"/> vector of Matrix. </remarks>
	void setClassCount(const size_t classcount) override;

	/// \copydoc IMatrixClassifier::train(const std::vector<std::vector<Eigen::MatrixXd>>&)
	bool train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets) override;

	/// \copydoc IMatrixClassifier::classify(const Eigen::MatrixXd&, size_t&)
	bool classify(const Eigen::MatrixXd& sample, size_t& classid) override;

	/// \copybrief IMatrixClassifier::classify(const Eigen::MatrixXd&, size_t&, std::vector<double>&, std::vector<double>&)
	/// <summary>	Compute the distance between the sample and each mean matrix.\n
	/// The class with the closest mean is the predicted class.\n
	/// The distance is returned.\n
	/// The probability \f$ \mathcal{P}_i \f$ to be the class \f$ i \f$ is compute as :
	/// \f[
	/// p_i = \frac{d_{\text{min}}}{d_i}\\
	/// \mathcal{P}_i =  \frac{p_i}{\sum{\left(p_i\right)}}
	/// \f]\n
	/// <b>Remark</b> : The probability is normalized \f$ \sum{\left(\mathcal{P}_i\right)} = 1 \f$
	///	</summary>
	/// \copydetails IMatrixClassifier::classify(const Eigen::MatrixXd&, size_t&, std::vector<double>&, std::vector<double>&)
	bool classify(const Eigen::MatrixXd& sample, size_t& classid, std::vector<double>& distance, std::vector<double>& probability) override;

	//***********************
	//***** XML Manager *****
	//***********************

	/// \copydoc IMatrixClassifier::saveXML(const std::string&)
	bool saveXML(const std::string& filename) override;

	/// \copydoc IMatrixClassifier::loadXML(const std::string&)
	bool loadXML(const std::string& filename) override;

	/// <summary>	Add the attribute on the first node.
	///
	/// -# The type of the classifier : MDM
	/// -# The number of class : <see cref="m_ClassCount"/>
	/// -# The metric to use : <see cref="m_Metric"/>
	/// </summary>
	/// \copydetails IMatrixClassifier::saveHeaderAttribute(tinyxml2::XMLElement*) const
	bool saveHeaderAttribute(tinyxml2::XMLElement* element) const override;

	/// <summary>	Loads the attribute on the first node.
	///
	/// -# Check the type : MDM
	/// -# The number of class : <see cref="m_ClassCount"/>
	/// -# The metric to use : <see cref="m_Metric"/>
	/// </summary>
	/// \copydetails IMatrixClassifier::loadHeaderAttribute(tinyxml2::XMLElement*)
	bool loadHeaderAttribute(tinyxml2::XMLElement* element) override;

	//*****************************
	//***** Override Operator *****
	//*****************************
	/// \copydoc IMatrixClassifier::isEqual(const IMatrixClassifier&, const double) const
	bool isEqual(const CMatrixClassifierMDM& obj, const double precision = 1e-6) const;
	
	/// \copydoc IMatrixClassifier::copy(const IMatrixClassifier&)
	void copy(const CMatrixClassifierMDM& obj);

	/// \copybrief IMatrixClassifier::getType()
	/// <returns>	Minimum Distance to Mean. </returns>
	std::string getType() const override { return "Minimum Distance to Mean"; }

	/// \copydoc IMatrixClassifier::print()
	std::stringstream print() const override;

	/// \copydoc IMatrixClassifier::operator=(const IMatrixClassifier&)
	CMatrixClassifierMDM& operator=(const CMatrixClassifierMDM& obj)
	{
		copy(obj);
		return *this;
	}

	/// \copydoc IMatrixClassifier::operator==(const IMatrixClassifier&) const
	bool operator==(const CMatrixClassifierMDM& obj) const { return isEqual(obj); }

	/// \copydoc IMatrixClassifier::operator!=(const IMatrixClassifier&) const
	bool operator!=(const CMatrixClassifierMDM& obj) const { return !isEqual(obj); }

	/// \copydoc IMatrixClassifier::operator<<(std::ostream&, const IMatrixClassifier&)
	friend std::ostream& operator <<(std::ostream& os, const CMatrixClassifierMDM& obj)
	{
		os << obj.print().str();
		return os;
	}
};
