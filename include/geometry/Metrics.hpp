///-------------------------------------------------------------------------------------------------
/// 
/// \file Metrics.hpp
/// \brief All Metrics.
/// \author Thibaut Monseigne (Inria).
/// \version 1.0.
/// \date 26/10/2018.
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// \remarks 
/// - List of Metrics inspired by the work of Alexandre Barachant : <a href="https://github.com/alexandrebarachant/pyRiemann">pyRiemann</a> (<a href="https://github.com/alexandrebarachant/pyRiemann/blob/master/LICENSE">License</a>).
/// 
///-------------------------------------------------------------------------------------------------

#pragma once
#include <string>

namespace Geometry {

/// <summary>	Enumeration of metrics. Inspired by the work of Alexandre Barachant : <a href="https://github.com/alexandrebarachant/pyRiemann">pyRiemann</a>. </summary>
enum class EMetric
{
	Riemann,		///< The Riemannian Metric.
	Euclidian,		///< The Euclidian Metric.
	LogEuclidian,	///< The Log Euclidian Metric.
	LogDet,			///< The Log Determinant Metric.
	Kullback,		///< The Kullback Metric.
	ALE,			///< The AJD-based log-Euclidean (ALE) Metric.
	Harmonic,		///< The Harmonic Metric.
	Wasserstein,	///< The Wasserstein Metric.
	Identity		///< The Identity Metric.
};

/// <summary>	Convert Metrics to string. </summary>
/// <param name="metric">	The metric. </param>
/// <returns>	std::string </returns>
inline std::string toString(const EMetric metric)
{
	switch (metric)
	{
		case EMetric::Riemann: return "Riemann";
		case EMetric::Euclidian: return "Euclidian";
		case EMetric::LogEuclidian: return "Log Euclidian";
		case EMetric::LogDet: return "Log Determinant";
		case EMetric::Kullback: return "Kullback";
		case EMetric::ALE: return "AJD-based log-Euclidean";
		case EMetric::Harmonic: return "Harmonic";
		case EMetric::Wasserstein: return "Wasserstein";
		case EMetric::Identity: return "Identity";
		default: return "Invalid Metric";
	}
}

/// <summary>	Convert string to Metric. </summary>
/// <param name="metric">	The metric. </param>
/// <returns>	<see cref="EMetric"/> </returns>
inline EMetric StringToMetric(const std::string& metric)
{
	if (metric == "Riemann") { return EMetric::Riemann; }
	if (metric == "Euclidian") { return EMetric::Euclidian; }
	if (metric == "Log Euclidian") { return EMetric::LogEuclidian; }
	if (metric == "Log Determinant") { return EMetric::LogDet; }
	if (metric == "Kullback") { return EMetric::Kullback; }
	if (metric == "AJD-based log-Euclidean") { return EMetric::ALE; }
	if (metric == "Harmonic") { return EMetric::Harmonic; }
	if (metric == "Wasserstein") { return EMetric::Wasserstein; }
	return EMetric::Identity;
}

}  // namespace Geometry
