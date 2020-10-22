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

#include "geometry/classifier/IMatrixClassifier.hpp"
#include "geometry/Metrics.hpp"

namespace Geometry {

/// <summary>	Class of Minimum Distance to Mean (MDM) Classifier. </summary>
/// <seealso cref="IMatrixClassifier" />
class CMatrixClassifierMDM : public IMatrixClassifier
{
public:
	//***********************	
	//***** Constructor *****
	//***********************	
	/// <summary>	Default constructor. Initializes a new instance of the <see cref="CMatrixClassifierMDM"/> class. </summary>
	CMatrixClassifierMDM() { CMatrixClassifierMDM::setClassCount(m_nbClass); }

	/// <summary>	Default Copy constructor. Initializes a new instance of the <see cref="CMatrixClassifierMDM"/> class. </summary>
	/// <param name="obj">	Initial object. </param>
	CMatrixClassifierMDM(const CMatrixClassifierMDM& obj) { *this = obj; }

	/// <summary>	Initializes a new instance of the <see cref="CMatrixClassifierMDM"/> class and set base members. </summary>
	/// \copydetails IMatrixClassifier(const size_t, const EMetric)
	explicit CMatrixClassifierMDM(size_t nbClass, EMetric metric);

	/// <summary>	Finalizes an instance of the <see cref="CMatrixClassifierMDM"/> class. </summary>
	/// <remarks>	clear the <see cref="m_means"/> vector of Matrix. </remarks>
	~CMatrixClassifierMDM() override;

	//***************************
	//***** Getter / Setter *****
	//***************************
	const std::vector<Eigen::MatrixXd>& getMeans() const { return m_means; }
	void setMeans(const std::vector<Eigen::MatrixXd>& means) { m_means = means; }

	const std::vector<size_t>& getTrialNumbers() const { return m_nbTrials; }
	void setTrialNumbers(const std::vector<size_t>& nbTrials) { m_nbTrials = nbTrials; }

	//**********************
	//***** Classifier *****
	//**********************
	/// \copydoc IMatrixClassifier::setClassCount(const size_t)
	/// <remarks>	resize the <see cref="m_means"/> vector of Matrix. </remarks>
	void setClassCount(size_t nbClass) override;

	/// \copybrief IMatrixClassifier::train(const std::vector<std::vector<Eigen::MatrixXd>>&)
	/// <summary>	
	/// -# Set the good number of classes
	/// -# Compute the mean of each class (row) with the metric (<see cref="EMetric" />) in <see cref="m_Metric"/> member.
	/// -# Set the number of trials for each class.
	///	</summary>
	/// \copydetails IMatrixClassifier::train(const std::vector<std::vector<Eigen::MatrixXd>>&)
	bool train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets) override;

	/// \copybrief IMatrixClassifier::classify(const Eigen::MatrixXd&, size_t&, std::vector<double>&, std::vector<double>&, const EAdaptations, const size_t&)
	/// <summary>	Compute the distance between the sample and each mean matrix.
	/// The class with the closest mean is the predicted class.\n
	/// The distances are returned.\n
	/// The probability \f$ \mathcal{P}_i \f$ to be the class \f$ i \f$ is compute as :
	/// \f[
	/// p_i = \frac{d_{\text{min}}}{d_i}\\
	/// \mathcal{P}_i =  \frac{p_i}{\sum{\left(p_i\right)}}
	/// \f]\n
	/// <b>Remark</b> : The probability is normalized \f$ \sum{\left(\mathcal{P}_i\right)} = 1 \f$\n
	/// If the classfier is adapted, launch adaptation method (expected class if supervised, predicted class if unsupervised).\n 
	/// With \f$ C_k \f$ the prototype (mean) of the Class \f$ k \f$, \f$ \gamma_m \f$ the Geodesic (<see cref="Geodesic" />) with the metric \f$ m \f$ (<see cref="EMetric" />), 
	/// \f$ S \f$ the current trial (sample) and \f$ N_k \f$ the number of trials for the class \f$ k \f$ (with the current trial).
	/// \f[
	/// C_k = \gamma_m\left( C_k,S,\frac{1}{N_k}\right)
	/// \f]
	///	</summary>
	/// \copydetails IMatrixClassifier::classify(const Eigen::MatrixXd&, size_t&, std::vector<double>&, std::vector<double>&, const EAdaptations, const size_t&)
	bool classify(const Eigen::MatrixXd& sample, size_t& classId, std::vector<double>& distance, std::vector<double>& probability,
				  EAdaptations adaptation = EAdaptations::None, const size_t& realClassId = std::numeric_limits<size_t>::max()) override;

	//*****************************
	//***** Override Operator *****
	//*****************************

	/// \copydoc IMatrixClassifier::isEqual(const IMatrixClassifier&, const double) const
	bool isEqual(const CMatrixClassifierMDM& obj, double precision = 1e-6) const;

	/// \copydoc IMatrixClassifier::copy(const IMatrixClassifier&)
	void copy(const CMatrixClassifierMDM& obj);

	/// \copybrief IMatrixClassifier::getType()
	/// <returns>	Minimum Distance to Mean. </returns>
	std::string getType() const override { return toString(EMatrixClassifiers::MDM); }

	/// <summary>	Override the affectation operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	The copied object. </returns>
	CMatrixClassifierMDM& operator=(const CMatrixClassifierMDM& obj)
	{
		copy(obj);
		return *this;
	}

	/// <summary>	Override the egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	<c>True</c> if the two <see cref="CMatrixClassifierMDM"/> are equals. </returns>
	bool operator==(const CMatrixClassifierMDM& obj) const { return isEqual(obj); }

	/// <summary>	Override the not egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	<c>True</c> if the two <see cref="CMatrixClassifierMDM"/> are diffrents. </returns>
	bool operator!=(const CMatrixClassifierMDM& obj) const { return !isEqual(obj); }

	/// <summary>	Override the ostream operator. </summary>
	/// <param name="os">	The ostream. </param>
	/// <param name="obj">	The object. </param>
	/// <returns>	Return the modified ostream. </returns>
	friend std::ostream& operator <<(std::ostream& os, const CMatrixClassifierMDM& obj)
	{
		os << obj.print().str();
		return os;
	}

protected:
	//***********************
	//***** XML Manager *****
	//***********************
	/// <summary>	Save Classes informations (Mean and number of trials of each class). </summary>
	/// <returns>	<c>True</c> if it succeeds, <c>false</c> otherwise. </returns>
	bool saveClasses(tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* data) const override;

	/// <summary>	Load Classes informations (Mean and number of trials of each class). </summary>
	/// <returns>	<c>True</c> if it succeeds, <c>false</c> otherwise. </returns>
	bool loadClasses(tinyxml2::XMLElement* data) override;

	//*****************************
	//***** Override Operator *****
	//*****************************
	std::stringstream printClasses() const override;

	//*********************
	//***** Variables *****
	//*********************
	std::vector<Eigen::MatrixXd> m_means;	///< Mean Matrix of each class.
	std::vector<size_t> m_nbTrials;			///< Number of trials of each class.
};

}  // namespace Geometry
