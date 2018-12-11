///-------------------------------------------------------------------------------------------------
/// 
/// \file CFeatureClassifierLDA.hpp
/// 
/// \brief Linear Discriminant Analysis (LDA) Classifier.
/// 
/// \author Thibaut Monseigne (Inria).
/// 
/// \version 1.0.
/// 
/// \date 11/12/2018.
/// 
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// 
/// \remarks 
/// - <a href="https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html">Linear Discriminant Analysis (LDA)</a> inspired by <a href="http://scikit-learn.org">sklearn</a> (<a href="https://github.com/scikit-learn/scikit-learn/blob/master/COPYING">License</a>).
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include "IFeatureClassifier.hpp"
#include <Eigen/Dense>
#include <vector>

class CFeatureClassifierLDA : public IFeatureClassifier
{
public:
	size_t m_DimensionCount = 0,
		   m_FeatureCount = 0;
	//***********************	
	//***** Constructor *****
	//***********************	
	CFeatureClassifierLDA() = default;
	~CFeatureClassifierLDA() = default;

	//**********************
	//***** Classifier *****
	//**********************
	bool train(const std::vector<std::vector<Eigen::RowVectorXd>>& datasets) override;
	
	bool classify(const Eigen::RowVectorXd& sample, size_t& classid) override;
	bool classify(const Eigen::RowVectorXd& sample, size_t& classid, std::vector<double>& distance, std::vector<double>& probability) override;

	//***********************
	//***** XML Manager *****
	//***********************

	//*****************************
	//***** Override Operator *****
	//*****************************

};
