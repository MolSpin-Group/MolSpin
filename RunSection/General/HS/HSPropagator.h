/////////////////////////////////////////////////////////////////////////
// HSPropagator (RunSection::General::HS)
// ------------------
// Numerical propagation strategies for Hilbert factors and density matrices.
// This component owns time stepping, pulse-segment propagation, and the static
// time-infinity solve; TaskHSGeneral only orchestrates these operations.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_HS_HSPropagator
#define MOD_RunSection_General_HS_HSPropagator

#include <armadillo>
#include <string>
#include "SpinSpace.h"
#include "HSExecutionPlan.h"
#include "HSOrientationSampler.h"
#include "SpinAPIfwd.h"
#include <iosfwd>
#include <functional>
#include <tuple>
#include <vector>

namespace RunSection::General::HS
{
	class HSReactionRelaxation;
	struct HSRelaxationContext;

	using HSPulseTimelineObserver = std::function<bool(double, const arma::cx_mat &, bool, std::string &)>;

	class HSPropagator
	{
	public:
		HSPropagator(const HSExecutionPlan &_plan, SpinAPI::SpinSpace &_space)
			: plan(_plan), space(_space) {}

		bool Step(const arma::sp_cx_mat &_hamiltonian, const arma::sp_cx_mat &_reaction,
			double _dt, arma::cx_mat &_factors, std::string &_error);
		bool StepDensity(const arma::sp_cx_mat &_hamiltonian, const arma::sp_cx_mat &_reaction,
			double _dt, arma::cx_mat &_density, const HSReactionRelaxation &_relaxation,
			const HSRelaxationContext &_context, std::string &_error);
		bool StepDensitySplit(const arma::sp_cx_mat &_hamiltonian, const arma::sp_cx_mat &_reaction,
			double _dt, arma::cx_mat &_density, const HSReactionRelaxation &_relaxation,
			const HSRelaxationContext &_context, std::string &_error);
		bool StepDensityDynamicRK4(const arma::sp_cx_mat &_hamiltonianStart,
			const arma::sp_cx_mat &_reactionStart, const arma::sp_cx_mat &_hamiltonianMid,
			const arma::sp_cx_mat &_reactionMid, const arma::sp_cx_mat &_hamiltonianEnd,
			const arma::sp_cx_mat &_reactionEnd, double _dt, arma::cx_mat &_density,
			const HSReactionRelaxation &_relaxation, const HSRelaxationContext &_context,
			std::string &_error);

		bool SolveTimeInfinity(const arma::sp_cx_mat &_hamiltonian,
			const arma::sp_cx_mat &_reaction, const arma::cx_mat &_initialDensity,
			const HSReactionRelaxation &_relaxation, const HSRelaxationContext &_context,
			arma::cx_mat &_integratedDensity, std::string &_error);

		bool ApplyPulsePreparationSequence(
			const std::vector<std::tuple<std::string, double>> &_sequence,
			const SpinAPI::system_ptr &_system, const HSOrientation &_orientation,
			const arma::sp_cx_mat &_baseHamiltonian, const arma::sp_cx_mat &_baseReaction,
			const HSReactionRelaxation &_relaxation, const HSRelaxationContext &_context,
			bool _densityMode, arma::cx_mat &_factors, arma::cx_mat &_density,
			double &_elapsedTime, const HSPulseTimelineObserver &_observer,
			std::ostream &_log, std::string &_error);

	private:
		const HSExecutionPlan &plan;
		SpinAPI::SpinSpace &space;
		arma::mat highamWorkspace;
	};
}
#endif
