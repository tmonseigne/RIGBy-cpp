///-------------------------------------------------------------------------------------------------
/// 
/// \file IFeatureClassifier.hpp
/// 
/// \brief Abstract class of Feature Classifier
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

///----------------------------------------------------------------------------------------------------
///
/// <summary>	Abstract class of Feature Classifier. </summary>
///
///----------------------------------------------------------------------------------------------------
class IFeatureClassifier
{
public:

	//***********************
	//***** Constructor *****
	//***********************
	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Initializes a new instance of the <see cref="IFeatureClassifier"/> class. </summary>
	///
	///----------------------------------------------------------------------------------------------------
	IFeatureClassifier() = default;

	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Finalizes an instance of the <see cref="IMatrixClassifier"/> class. </summary>
	///
	///----------------------------------------------------------------------------------------------------
	virtual ~IFeatureClassifier() = default;

	//**********************
	//***** Classifier *****
	//**********************
	
	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Train the classifier with the dataset. </summary>
	///
	/// <param name="datasets">	The data set one class by row and trials on colums. </param>
	///
	/// <returns>	True if it succeeds, false if it fails. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	virtual bool train(const std::vector<std::vector<Eigen::RowVectorXd>>& datasets) = 0;

	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Classify the feature and return the class id. </summary>
	///
	/// <param name="sample">	The sample to classify. </param>
	/// <param name="classid">	The predicted class. </param>
	///
	/// <returns>	True if it succeeds, false if it fails. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	virtual bool classify(const Eigen::RowVectorXd& sample, size_t& classid) = 0;

	///----------------------------------------------------------------------------------------------------
	///
	/// <summary>	Classify the feature and return the class id, the distance and the probability of each class. </summary>
	///
	/// <param name="sample">		The sample to classify. </param>
	/// <param name="classid">		The predicted class. </param>
	/// <param name="distance">		The distance of the sample with each class. </param>
	/// <param name="probability">	The distance of the sample with each class. </param>
	///
	/// <returns>	True if it succeeds, false if it fails. </returns>
	///
	///----------------------------------------------------------------------------------------------------
	virtual bool classify(const Eigen::RowVectorXd& sample, size_t& classid, std::vector<double>& distance, std::vector<double>& probability) = 0;

	//***********************
	//***** XML Manager *****
	//***********************

};
