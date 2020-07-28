///-------------------------------------------------------------------------------------------------
/// 
/// \file CASR.hpp
/// \brief Class used to add Rebias to Other Classifier.
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 27/08/2019.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// 
///-------------------------------------------------------------------------------------------------
#pragma once

/*
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <map>


#include "3rd-party/tinyxml2.h"

class CASR
{
public:

	static const std::map<size_t, std::pair<int, int>> Filters;
	
	CASR() = default;	///< Initializes a new instance of the <see cref="CASR"/> class.
	~CASR() = default;	///< Finalizes an instance of the <see cref="CASR"/> class.

	void computeStat(const Eigen::MatrixXd& data);
	
	void train(const std::vector<Eigen::MatrixXd>& dataset, const double rejectionLimit = 5);

	bool process(const Eigen::MatrixXd& in, Eigen::MatrixXd& out);

	
	//***********************
	//***** XML Manager *****
	//***********************
	/// <summary>	Saves the Bias information in an XML file. </summary>
	/// <param name="filename">	Filename. </param>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	bool saveXML(const std::string& filename) const;

	/// <summary>	Loads the Bias information from an XML file. </summary>
	/// <param name="filename">	Filename. </param>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	bool loadXML(const std::string& filename);

	/// <summary>	Save informations in xml element (Bias and number of classification). </summary>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	bool saveAdditional(tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* data) const;

	/// <summary>	Load informations in xml element (Bias and number of classification). </summary>
	/// <returns>	True if it succeeds, false if it fails. </returns>
	bool loadAdditional(tinyxml2::XMLElement* data);

	//*****************************
	//***** Override Operator *****
	//*****************************
	bool isEqual(const CASR& obj, const double precision = 1e-6) const;

	/// <summary>	Copy object value. </summary>
	/// <param name="obj">	The object to copy. </param>
	void copy(const CASR& obj);

	/// <summary>	Get the Classifier information for output. </summary>
	/// <returns>	The Classifier print in stringstream. </returns>
	std::string toString() const;

	/// <summary>	Override the affectation operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	The copied object. </returns>
	CASR& operator=(const CASR& obj)
	{
		copy(obj);
		return *this;
	}

	/// <summary>	Override the egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	True if the two <see cref="CASR"/> are equals. </returns>
	bool operator==(const CASR& obj) const { return isEqual(obj); }

	/// <summary>	Override the not egal operator. </summary>
	/// <param name="obj">	The second object. </param>
	/// <returns>	True if the two <see cref="CASR"/> are diffrents. </returns>
	bool operator!=(const CASR& obj) const { return !isEqual(obj); }

	/// <summary>	Override the ostream operator. </summary>
	/// <param name="os">	The ostream. </param>
	/// <param name="obj">	The object. </param>
	/// <returns>	Return the modified ostream. </returns>
	friend std::ostream& operator <<(std::ostream& os, const CASR& obj)
	{
		os << obj.toString();
		return os;
	}

	//*********************
	//***** Variables *****
	//*********************
};
*/