#pragma once
#include <ostream>
#include "utils/Metrics.hpp"
#include <vector>
#include <Eigen/Dense>

class CMatrixClassifierFgDA
{
public:
	EMetrics m_Metric = Metric_Riemann;

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

