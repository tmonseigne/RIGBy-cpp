///-------------------------------------------------------------------------------------------------
/// 
/// \file Metrics.hpp
/// 
/// \brief All Metrics.
/// 
/// \author Thibaut Monseigne (Inria).
/// 
/// \version 1.0.
/// 
/// \date 26/10/2018.
/// 
/// \copyright <a href="https://choosealicense.com/licenses/agpl-3.0/">GNU Affero General Public License v3.0</a>.
/// 
/// \remarks 
/// - List of Metrics inspired by the work of Alexandre Barachant : <a href="https://github.com/alexandrebarachant/pyRiemann">pyRiemann</a> (<a href="https://github.com/alexandrebarachant/pyRiemann/blob/master/LICENSE">License</a>).
/// 
///-------------------------------------------------------------------------------------------------

#pragma once
#include <string>

/// <summary> Enumeration of metrics. Inspired by the work of Alexandre Barachant : <a href="https://github.com/alexandrebarachant/pyRiemann">pyRiemann</a>.</summary>
enum EMetrics
{
	/// <summary>	The Riemannian Metric. </summary>
	Metric_Riemann,
	/// <summary>	The Euclidian Metric. </summary>
	Metric_Euclidian,
	/// <summary>	The Log Euclidian Metric. </summary>
	Metric_LogEuclidian,
	/// <summary>	The Log Determinant Metric. </summary>
	Metric_LogDet,
	/// <summary>	The Kullback Metric. </summary>
	Metric_Kullback,
	/// <summary>	The AJD-based log-Euclidean (ALE) Metric. </summary>
	Metric_ALE,
	/// <summary>	The Harmonic Metric. </summary>
	Metric_Harmonic,
	/// <summary>	The Wasserstein Metric. </summary>
	Metric_Wasserstein,
	/// <summary>	The Identity Metric. </summary>
	Metric_Identity
};

inline std::string MetricToString(const EMetrics metric)
{
	switch (metric)
	{
		case Metric_Riemann: return "Riemann";
		case Metric_Euclidian: return "Euclidian";
		case Metric_LogEuclidian: return "Log Euclidian";
		case Metric_LogDet: return "Log Determinant";
		case Metric_Kullback: return "Kullback";
		case Metric_ALE: return "AJD-based log-Euclidean";
		case Metric_Harmonic: return "Harmonic";
		case Metric_Wasserstein: return "Wasserstein";
		case Metric_Identity: return "Identity";
		default: return "Invalid";
	}
}
