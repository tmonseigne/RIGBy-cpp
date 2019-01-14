///-------------------------------------------------------------------------------------------------
/// 
/// \file CMatrixClassifierFgMDM.hpp
/// \brief Class of Minimum Distance to Mean with geodesic filtering (FgMDM) Classifier
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 10/12/2018.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
///
///-------------------------------------------------------------------------------------------------

#pragma once

#include "CMatrixClassifierMDM.hpp"

/// <summary>	Class of Minimum Distance to Mean with geodesic filtering (FgMDM) Classifier. </summary>
/// <seealso cref="CMatrixClassifierMDM" />
class CMatrixClassifierFgMDM : public CMatrixClassifierMDM
{
public:
	Eigen::MatrixXd m_Ref, m_Weight;

	//***********************	
	//***** Constructor *****
	//***********************	
	/// <summary>	Initializes a new instance of the <see cref="CMatrixClassifierFgMDM"/> class. </summary>
	CMatrixClassifierFgMDM() = default;

	/// <summary>	Default Copy constructor. Initializes a new instance of the <see cref="CMatrixClassifierMDM"/> class. </summary>
	/// \copydetails IMatrixClassifier(const IMatrixClassifier&)
	CMatrixClassifierFgMDM(const CMatrixClassifierFgMDM& obj);

	/// <summary>	Initializes a new instance of the <see cref="CMatrixClassifierFgMDM"/> class and set base members. </summary>
	/// \copydetails IMatrixClassifier(size_t, EMetrics)
	explicit CMatrixClassifierFgMDM(const size_t classcount, const EMetrics metric) : CMatrixClassifierMDM(classcount, metric) { }

	/// <summary>	Finalizes an instance of the <see cref="CMatrixClassifierFgMDM"/> class. </summary>
	/// <remarks>	clear the <see cref="m_Means"/> vector of Matrix. </remarks>
	~CMatrixClassifierFgMDM() override = default;

	//**********************
	//***** Classifier *****
	//**********************
	/// \copydoc IMatrixClassifier::train(const std::vector<std::vector<Eigen::MatrixXd>>&)
	bool train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets) override;

	/// \copydoc IMatrixClassifier::classify(const Eigen::MatrixXd&, size_t&)
	bool classify(const Eigen::MatrixXd& sample, size_t& classid) override;

	/// \copybrief IMatrixClassifier::classify(const Eigen::MatrixXd&, size_t&, std::vector<double>&, std::vector<double>&)
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
	/// -# The type of the classifier : FgMDM
	/// -# The number of class : <see cref="m_ClassCount"/>
	/// -# The metric to use : <see cref="m_Metric"/>
	/// </summary>
	/// \copydetails IMatrixClassifier::saveHeaderAttribute(tinyxml2::XMLElement*) const
	bool saveHeaderAttribute(tinyxml2::XMLElement* element) const override;

	/// <summary>	Loads the attribute on the first node.
	///
	/// -# Check the type : FgMDM
	/// -# The number of class : <see cref="m_ClassCount"/>
	/// -# The metric to use : <see cref="m_Metric"/>
	/// </summary>
	/// \copydetails IMatrixClassifier::loadHeaderAttribute(tinyxml2::XMLElement*)
	bool loadHeaderAttribute(tinyxml2::XMLElement* element) override;

	//*****************************
	//***** Override Operator *****
	//*****************************
	/// \copydoc IMatrixClassifier::isEqual(const IMatrixClassifier&, const double) const
	bool isEqual(const CMatrixClassifierFgMDM& obj, double precision = 1e-6) const;

	/// \copydoc IMatrixClassifier::copy(const IMatrixClassifier&)
	void copy(const CMatrixClassifierFgMDM& obj);

	/// \copybrief IMatrixClassifier::getType()
	/// <returns>	Minimum Distance to Mean with geodesic filtering. </returns>
	std::string getType() const override { return "Minimum Distance to Mean with geodesic filtering"; }
	
	/// \copydoc IMatrixClassifier::print()
	std::stringstream print() const override;

	/// \copydoc IMatrixClassifier::operator=(const IMatrixClassifier&)
	CMatrixClassifierFgMDM& operator=(const CMatrixClassifierFgMDM& obj)
	{
		copy(obj);
		return *this;
	}

	/// \copydoc IMatrixClassifier::operator==(const IMatrixClassifier&) const
	bool operator==(const CMatrixClassifierFgMDM& obj) const { return isEqual(obj); }

	/// \copydoc IMatrixClassifier::operator!=(const IMatrixClassifier&) const
	bool operator!=(const CMatrixClassifierFgMDM& obj) const { return !isEqual(obj); }

	/// \copydoc IMatrixClassifier::operator<<(std::ostream&, const IMatrixClassifier&)
	friend std::ostream& operator <<(std::ostream& os, const CMatrixClassifierFgMDM& obj)
	{
		os << obj.print().str();
		return os;
	}
};
