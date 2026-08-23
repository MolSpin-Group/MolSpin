/////////////////////////////////////////////////////////////////////////
// HSObservableCollector (RunSection::General::HS)
// ------------------
// Defines Hilbert-space observables and owns observable post-processing:
// finite-time quadrature, segment-aware cumulative integration, and the legacy
// finite-time yield correction. It does not own RunSection output streams.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_HS_HSObservableCollector
#define MOD_RunSection_General_HS_HSObservableCollector

#include <armadillo>
#include <iosfwd>
#include <string>
#include <vector>
#include "SpinAPIfwd.h"
#include "SpinSpace.h"
#include "HSExecutionPlan.h"
#include "HSOrientationSampler.h"

namespace RunSection::General::HS
{
	struct HSObservable
	{
		std::string name;

		// A state/source projector may rotate with a molecular-frame powder
		// orientation.  A spin-polarization operator is deliberately kept in
		// the laboratory frame.  CIDSP/CIDNP observables combine both pieces as
		// I_alpha * P_source after rotating only P_source.
		arma::sp_cx_mat stateOrSource;
		SpinAPI::HilbertStateRotationCache stateRotationCache;
		bool hasStateOrSource = false;
		bool rotateStateOrSource = false;

		arma::sp_cx_mat fixedSpinOperator;
		bool hasFixedSpinOperator = false;

		SpinAPI::transition_ptr transition;
	};

	class HSObservableCollector
	{
	public:
		bool Prepare(const HSExecutionPlan &_plan, const SpinAPI::system_ptr &_system,
			SpinAPI::SpinSpace &_space, std::ostream &_log, std::string &_error);
		bool OperatorsForOrientation(SpinAPI::SpinSpace &_space, const HSOrientation &_orientation,
			std::vector<arma::sp_cx_mat> &_operators, std::string &_error) const;
		void Evaluate(const std::vector<arma::sp_cx_mat> &_operators, const arma::cx_mat &_factors,
			arma::rowvec &_values) const;
		void EvaluateDensity(const std::vector<arma::sp_cx_mat> &_operators, const arma::cx_mat &_density,
			arma::rowvec &_values) const;
		bool IntegrateFiniteTime(const arma::mat &_values, double _timeStep,
			arma::rowvec &_integrated, std::string &_error) const;
		bool IntegrateTimeline(const std::vector<double> &_times, const arma::mat &_values,
			bool _enabled, arma::mat &_output, std::string &_error) const;
		bool ApplyFiniteTimeYieldCorrection(const HSExecutionPlan &_plan,
			const SpinAPI::system_ptr &_system, SpinAPI::SpinSpace &_space,
			arma::rowvec &_integrated, std::ostream &_log, std::string &_error) const;
		void WriteHeader(const SpinAPI::system_ptr &_system, std::ostream &_stream) const;
		size_t Size() const { return observables.size(); }
		const std::vector<HSObservable> &Observables() const { return observables; }

	private:
		std::vector<HSObservable> observables;
	};
}
#endif
