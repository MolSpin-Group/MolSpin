/////////////////////////////////////////////////////////////////////////
// HSStatePreparation (RunSection::General::HS)
// ------------------
// Constructs normalized Hilbert-space initial states and owns every
// orientation-dependent state-preparation operation used by TaskHSGeneral.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_HS_HSStatePreparation
#define MOD_RunSection_General_HS_HSStatePreparation

#include <armadillo>
#include <iosfwd>
#include <random>
#include <string>
#include <vector>

#include "SpinAPIfwd.h"
#include "SpinSpace.h"
#include "HSExecutionPlan.h"
#include "HSOrientationSampler.h"

namespace RunSection::General::HS
{
	// Reference state prepared once per SpinSystem. Direct calculations keep a
	// normalized density/factorization; stochastic calculations additionally
	// keep the sampled Hilbert factors. Molecular-frame rotation generators are
	// cached here so the powder loop never reconstructs state-rotation operators.
	struct HSPreparedState
	{
		arma::cx_mat density;
		arma::cx_mat factors;
		SpinAPI::HilbertTraceSampleSet traceSamples;
		SpinAPI::StateFrame frame = SpinAPI::StateFrame::Fixed;
		SpinAPI::HilbertStateRotationCache rotationCache;
		std::vector<std::string> dephasingHamiltonian;
		bool stochastic = false;
		bool hasRotationCache = false;
		bool dephaseInHamiltonianEigenbasis = false;
	};

	// Orientation-specific state used by the propagation engine. The density is
	// always available; factors satisfy rho = B B^dagger and are used whenever
	// the calculation does not require a general density-matrix dissipator.
	struct HSOrientedState
	{
		arma::cx_mat density;
		arma::cx_mat factors;
	};

	class HSStatePreparation
	{
	public:
		static bool ValidateTraceSampling(const SpinAPI::system_ptr &_system, std::string &_error);
		static bool BuildInitialDensity(const SpinAPI::system_ptr &_system, SpinAPI::SpinSpace &_space,
			arma::cx_mat &_density, std::string &_error);
		static bool Prepare(const HSExecutionPlan &_plan, const SpinAPI::system_ptr &_system,
			SpinAPI::SpinSpace &_space, HSPreparedState &_state,
			std::mt19937 &_generator, std::ostream &_log, std::string &_error);
		static bool PrepareForOrientation(const HSExecutionPlan &_plan, SpinAPI::SpinSpace &_space,
			const HSPreparedState &_reference, const HSOrientation &_orientation,
			HSOrientedState &_state, std::string &_error);
		static void SeedGenerator(const HSExecutionPlan &_plan, std::mt19937 &_generator, std::ostream &_log);
	};
}
#endif
