#pragma once

#include <Eigen/Dense>
#include <vector>
#include "utils/Metrics.hpp"

class CRebias
{
public:
	CRebias() = default;
	~CRebias() = default;

	bool computeRebias(const std::vector<std::vector<Eigen::MatrixXd>>& datasets, const EMetrics metric = Metric_Riemann);

	void applyRebias(const std::vector<std::vector<Eigen::MatrixXd>>& in, std::vector<std::vector<Eigen::MatrixXd>>& out);
	void applyRebias(const std::vector<Eigen::MatrixXd>& in, std::vector<Eigen::MatrixXd>& out);
	void applyRebias(const Eigen::MatrixXd& in, Eigen::MatrixXd& out);

	void updateRebias(const Eigen::MatrixXd& sample, const EMetrics metric = Metric_Riemann);

	//*****************************
	//***** Override Operator *****
	//*****************************
	/// \copydoc IMatrixClassifier::isEqual(const IMatrixClassifier&, const double) const
	bool isEqual(const CRebias& obj, const double precision = 1e-6) const;

	/// <summary>	Copy object value. </summary>
	/// <param name="obj">	The object to copy. </param>
	void copy(const CRebias& obj);

	/// <summary>	Get the Classifier information for output. </summary>
	/// <returns>	The Classifier print in stringstream. </returns>
	std::stringstream print() const;

	/// <summary>	Override the affectation operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	The copied object. </returns>
	CRebias& operator=(const CRebias& obj)
	{
		copy(obj);
		return *this;
	}

	/// <summary>	Override the egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	True if the two <see cref="CRebias"/> are equals. </returns>
	bool operator==(const CRebias& obj) const { return isEqual(obj); }

	/// <summary>	Override the not egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	True if the two <see cref="CRebias"/> are diffrents. </returns>
	bool operator!=(const CRebias& obj) const { return !isEqual(obj); }

	/// <summary>	Override the ostream operator. </summary>
	/// <param name="os">	The ostream. </param>
	/// <param name="obj">	The object. </param>
	/// <returns>	Return the modified ostream. </returns>
	friend std::ostream& operator <<(std::ostream& os, const CRebias& obj)
	{
		os << obj.print().str();
		return os;
	}

	//*********************
	//***** Variables *****
	//*********************
	/// <summary>	Rebias Matrix. </summary>
	Eigen::MatrixXd m_Bias, m_BiasIS;
	/// <summary>	Number of classification launched. </summary>
	size_t m_NClassify = 0;
};
