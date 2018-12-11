///-------------------------------------------------------------------------------------------------
/// 
/// \file CMatrixClassifierFgDA.hpp
/// 
/// \brief Fisher Geodesic Discriminant analysis Class.
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
/// - Inspired by the work of Alexandre Barachant : <a href="https://github.com/alexandrebarachant/pyRiemann">pyRiemann</a> (<a href="https://github.com/alexandrebarachant/pyRiemann/blob/master/LICENSE">License</a>).
/// 
///-------------------------------------------------------------------------------------------------

#pragma once

#include <ostream>
#include <vector>
#include <Eigen/Dense>
#include "utils/Metrics.hpp"
#include "CFeatureClassifierLDA.hpp"

class CMatrixClassifierFgDA
{
public:
	EMetrics m_Metric = Metric_Riemann;
	CFeatureClassifierLDA m_LDA;

	//***********************	
	//***** Constructor *****
	//***********************	
	CMatrixClassifierFgDA() = default;
	~CMatrixClassifierFgDA() = default;

	//********************
	//***** Computes *****
	//********************
	bool compute(std::vector<Eigen::MatrixXd>& dataset);
	bool filter(Eigen::MatrixXd& filter);
	
	//*****************************
	//***** Override Operator *****
	//*****************************
	bool operator==(const CMatrixClassifierFgDA& obj) const;
	bool operator!=(const CMatrixClassifierFgDA& obj) const;
	std::stringstream print() const;
	friend std::ostream& operator <<(std::ostream& os, const CMatrixClassifierFgDA& obj);
};

