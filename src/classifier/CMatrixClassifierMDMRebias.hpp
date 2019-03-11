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

#include "CMatrixClassifierMDM.hpp"
#include "utils/Metrics.hpp"

/// <summary>	Class of Minimum Distance to Mean (MDM) Classifier. </summary>
/// <seealso cref="IMatrixClassifier" />
class CMatrixClassifierMDMRebias : public CMatrixClassifierMDM
{
public:
	//***********************	
	//***** Constructor *****
	//***********************	
	/// <summary>	Default constructor. Initializes a new instance of the <see cref="CMatrixClassifierMDMRebias"/> class. </summary>
	CMatrixClassifierMDMRebias() = default;

	/// <summary>	Initializes a new instance of the <see cref="CMatrixClassifierMDMRebias"/> class and set base members. </summary>
	/// \copydetails CMatrixClassifierMDM(const size_t, const EMetrics)
	explicit CMatrixClassifierMDMRebias(const size_t classcount, const EMetrics metric) : CMatrixClassifierMDM(classcount, metric) { }

	/// <summary>	Finalizes an instance of the <see cref="CMatrixClassifierMDMRebias"/> class. </summary>
	/// \copydetails ~CMatrixClassifierMDM()
	virtual ~CMatrixClassifierMDMRebias() = default;

	//**********************
	//***** Classifier *****
	//**********************

	/// \copydoc CMatrixClassifierMDM::train(const std::vector<std::vector<Eigen::MatrixXd>>&)
	bool train(const std::vector<std::vector<Eigen::MatrixXd>>& datasets) override;

	/// \copybrief CMatrixClassifierMDM::classify(const Eigen::MatrixXd&, size_t&, std::vector<double>&, std::vector<double>&, const EAdaptations, const size_t&)
	/// <summary>	Compute the distance between the sample and each mean matrix.\n
	/// The class with the closest mean is the predicted class.\n
	/// The distance is returned.\n
	/// The probability \f$ \mathcal{P}_i \f$ to be the class \f$ i \f$ is compute as :
	/// \f[
	/// p_i = \frac{d_{\text{min}}}{d_i}\\
	/// \mathcal{P}_i =  \frac{p_i}{\sum{\left(p_i\right)}}
	/// \f]\n
	/// <b>Remark</b> : The probability is normalized \f$ \sum{\left(\mathcal{P}_i\right)} = 1 \f$\n
	/// If the classfier is adapted, launch adaptation method
	///	</summary>
	/// \copydetails IMatrixClassifier::classify(const Eigen::MatrixXd&, size_t&, std::vector<double>&, std::vector<double>&, const EAdaptations, const size_t&)
	bool classify(const Eigen::MatrixXd& sample, size_t& classId, std::vector<double>& distance, std::vector<double>& probability,
				  const EAdaptations adaptation = Adaptation_None, const size_t& realClassId = std::numeric_limits<std::size_t>::max()) override;

	/// <summary>	Classify the matrix and return the class id, the distance and the probability of each class.  
	///
	///  \f[ C^{k} = \gamma_\text{R}{ \left(C^{k},C_{sample},\frac{1}{N_{k}+1}\right) } \f]\n
	/// With :  \f$ k \f$ the class id, \f$ C^{k} \f$ the mean of the class, \f$ \gamma_\text{R} \f$ the Riemann geodesic (<see cref="Geodesic"/>),
	/// \f$ C_{sample} \f$ the current sample and \f$ N_{k} \f$ the number of trials for the class \f$k\f$
	/// </summary>
	/// <param name="sample">		The sample that adapts the classifier. </param>
	/// <param name="classid">		The class to adapt. </param>
	/// <returns>	True if it succeeds, false if it fails. </returns>


	//***********************
	//***** XML Manager *****
	//***********************
	/// \copydoc CMatrixClassifierMDM::saveXML(const std::string&)
	bool saveXML(const std::string& filename) override;

	/// \copydoc CMatrixClassifierMDM::loadXML(const std::string&)
	bool loadXML(const std::string& filename) override;

	//*****************************
	//***** Override Operator *****
	//*****************************

	/// \copydoc CMatrixClassifierMDM::isEqual(const CMatrixClassifierMDM&, const double) const
	bool isEqual(const CMatrixClassifierMDMRebias& obj, const double precision = 1e-6) const;

	/// \copydoc CMatrixClassifierMDM::copy(const CMatrixClassifierMDM&)
	void copy(const CMatrixClassifierMDMRebias& obj);

	/// \copybrief CMatrixClassifierMDM::getType()
	/// <returns>	Minimum Distance to Mean. </returns>
	std::string getType() const override { return "Minimum Distance to Mean REBIAS"; }

	/// \copydoc IMatrixClassifier::print()
	std::stringstream print() const override;

	/// <summary>	Override the affectation operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	The copied object. </returns>
	CMatrixClassifierMDMRebias& operator=(const CMatrixClassifierMDMRebias& obj) { copy(obj);		return *this; }

	/// <summary>	Override the egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	True if the two <see cref="CMatrixClassifierMDMRebias"/> are equals. </returns>
	bool operator==(const CMatrixClassifierMDMRebias& obj) const { return isEqual(obj); }

	/// <summary>	Override the not egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	True if the two <see cref="CMatrixClassifierMDMRebias"/> are diffrents. </returns>
	bool operator!=(const CMatrixClassifierMDMRebias& obj) const { return !isEqual(obj); }

	/// <summary>	Override the ostream operator. </summary>
	/// <param name="os">	The ostream. </param>
	/// <param name="obj">	The object. </param>
	/// <returns>	Return the modified ostream. </returns>
	friend std::ostream& operator <<(std::ostream& os, const CMatrixClassifierMDMRebias& obj) { os << obj.print().str();		return os; }


	//***** Variables *****
	/// <summary>	Rebias Matrix. </summary>
	Eigen::MatrixXd m_Rebias;
	/// <summary>	Number of classify launch. </summary>
	size_t m_NbClassify = 0;

protected:
	//***********************
	//***** XML Manager *****
	//***********************
	/// <summary>	Add the attribute on the first node.
	///
	/// -# The type of the classifier : MDM
	/// -# The number of classes : <see cref="m_ClassCount"/>
	/// -# The metric to use : <see cref="m_Metric"/>
	/// </summary>
	/// \copydetails CMatrixClassifierMDM::saveHeaderAttribute(tinyxml2::XMLElement*) const
	bool saveHeaderAttribute(tinyxml2::XMLElement* element) const override;

	/// <summary>	Loads the attribute on the first node.
	///
	/// -# Check the type : MDM
	/// -# The number of classes : <see cref="m_ClassCount"/>
	/// -# The metric to use : <see cref="m_Metric"/>
	/// </summary>
	/// \copydetails CMatrixClassifierMDM::loadHeaderAttribute(tinyxml2::XMLElement*)
	bool loadHeaderAttribute(tinyxml2::XMLElement* element) override;
};
