/////////////////////////////////////////////////////////////////////////
// HSStatePreparation implementation (RunSection::General::HS)
// ------------------
// Initial-state normalization, state-aware trace sampling, molecular-frame
// powder rotation, and optional orientation-specific eigenbasis dephasing.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// DEVELOPER WORKFLOW / OWNERSHIP MAP
// ----------------------------------------------------------------------
// HSGeneral initial-state preparation.
//
// What is done here:
//   - Constructs and normalizes the requested initial state/density for the current orientation.
//   - Handles thermal/eigen/molecular-frame semantics and stochastic trace-sampling preparation.
//   - Applies task-level preparation rules before free propagation begins.
//
// Connections to the General framework / SpinAPI:
//   - Uses SpinAPI State, SpinSpace thermal-state and rotation utilities.
//   - SSLiouvillianBuilder owns analogous one-manifold density preparation for SS/MultiSS.
//
// Why this ownership is used:
//   - State-frame handling is centralized so orientation does not get applied twice or omitted.
//   - Stochastic trace sampling is an estimator of the same quantum trace, not a different Hamiltonian model.
/////////////////////////////////////////////////////////////////////////

#include "HSStatePreparation.h"

#include "SpinSystem.h"
#include "State.h"

#include <cmath>
#include <numeric>

namespace RunSection::General::HS
{
	bool HSStatePreparation::ValidateTraceSampling(const SpinAPI::system_ptr &system, std::string &error)
	{
		error.clear();
		if (system == nullptr)
		{
			error = "stochastic HSGeneral cannot use a null spin system";
			return false;
		}

		const auto initialStates = system->InitialState();
		if (initialStates.size() != 1)
		{
			error = "spin system \"" + system->Name() +
				"\" must define exactly one initial State object for stochastic trace sampling";
			return false;
		}
		if (initialStates.front() == nullptr)
		{
			error = "spin system \"" + system->Name() +
				"\" uses a thermal initial state, which cannot be combined with stochastic trace sampling";
			return false;
		}
		if (!system->Operators().empty())
		{
			error = "spin system \"" + system->Name() +
				"\" contains explicit relaxation operators; general dissipators require density evolution and are intentionally incompatible with pure-state trace sampling";
			return false;
		}
		if (system->InitialStateCoherences() != SpinAPI::InitialStateCoherenceMode::Keep)
		{
			error = "spin system \"" + system->Name() +
				"\" requests initial-state dephasing, which cannot be represented by pure-state trace samples";
			return false;
		}
		if (system->InitialStateFrame() == SpinAPI::StateFrame::Eigen)
		{
			error = "spin system \"" + system->Name() +
				"\" uses frame=eigen, which cannot be represented by State-object trace samples";
			return false;
		}
		return true;
	}

	bool HSStatePreparation::BuildInitialDensity(const SpinAPI::system_ptr &system,
		SpinAPI::SpinSpace &space, arma::cx_mat &density, std::string &error)
	{
		error.clear();
		density.reset();
		if (system == nullptr)
		{
			error = "cannot construct an initial state for a null spin system";
			return false;
		}

		const auto initialStates = system->InitialState();
		if (initialStates.empty())
		{
			error = "spin system \"" + system->Name() + "\" does not define an initial state";
			return false;
		}

		std::vector<double> weights = system->Weights();
		if (weights.empty())
			weights.assign(initialStates.size(), 1.0 / static_cast<double>(initialStates.size()));
		else if (weights.size() == 1 && initialStates.size() == 1)
			weights[0] = 1.0;
		else if (weights.size() != initialStates.size())
		{
			error = "the number of initial-state weights does not match the number of initial states";
			return false;
		}

		for (double weight : weights)
		{
			if (!std::isfinite(weight) || weight < 0.0)
			{
				error = "initial-state weights must be finite and non-negative";
				return false;
			}
		}
		const double weightSum = std::accumulate(weights.begin(), weights.end(), 0.0);
		if (!(weightSum > 0.0))
		{
			error = "initial-state weights must have a positive sum";
			return false;
		}
		for (double &weight : weights)
			weight /= weightSum;

		density.zeros(space.HilbertSpaceDimensions(), space.HilbertSpaceDimensions());
		for (size_t index = 0; index < initialStates.size(); ++index)
		{
			arma::cx_mat component;
			if (initialStates[index] == nullptr)
			{
				if (!space.GetThermalState(space, system->Temperature(),
					system->ThermalHamiltonianList(), component))
				{
					error = "failed to construct the thermal initial state for spin system \"" +
						system->Name() + "\"";
					return false;
				}
			}
			else if (!space.GetState(initialStates[index], component))
			{
				error = "failed to construct initial State \"" + initialStates[index]->Name() +
					"\" for spin system \"" + system->Name() + "\"";
				return false;
			}

			const arma::cx_double componentTrace = arma::trace(component);
			if (!std::isfinite(std::real(componentTrace)) || std::abs(componentTrace) == 0.0)
			{
				error = "an initial-state component has an invalid trace";
				return false;
			}
			density += weights[index] * component / componentTrace;
		}
		return true;
	}

	void HSStatePreparation::SeedGenerator(const HSExecutionPlan &plan,
		std::mt19937 &generator, std::ostream &log)
	{
		if (plan.autoSeed)
		{
			std::random_device randomDevice;
			generator.seed(randomDevice());
			log << "Autoseed is on." << std::endl;
			return;
		}

		const double seed = plan.seed == 0.0 ? 1.0 : plan.seed;
		generator.seed(static_cast<std::mt19937::result_type>(seed));
		log << "Seed number is " << seed << "." << std::endl;
	}

	bool HSStatePreparation::Prepare(const HSExecutionPlan &plan,
		const SpinAPI::system_ptr &system, SpinAPI::SpinSpace &space,
		HSPreparedState &state, std::mt19937 &generator,
		std::ostream &log, std::string &error)
	{
		state = HSPreparedState();
		error.clear();
		state.frame = system != nullptr
			? system->InitialStateFrame() : SpinAPI::StateFrame::Fixed;

		// `frame=eigen` has a precise meaning only for a canonical thermal
		// density.  It is rebuilt for every crystallite from the user-selected
		// thermal Hamiltonian, rather than rotating one already diagonalized
		// density.  In general U exp(-beta H) U^dagger equals exp(-beta UHU^dagger),
		// but rebuilding is required when the propagation Hamiltonian is
		// secularized while the physical thermal Hamiltonian remains full.
		if (state.frame == SpinAPI::StateFrame::Eigen)
		{
			const auto initialStates = system->InitialState();
			if (initialStates.size() != 1 || initialStates.front() != nullptr)
			{
				error = "initial-state frame=eigen requires exactly one Thermal initial state";
				return false;
			}
			state.orientationSpecificThermal = true;
			state.thermalHamiltonian = system->ThermalHamiltonianList();
			state.thermalTemperature = system->Temperature();
		}

		if (plan.IsStochastic())
		{
			if (!ValidateTraceSampling(system, error))
				return false;

			SpinAPI::TraceSamplingMethod method = SpinAPI::TraceSamplingMethod::SUZ;
			if (plan.samplingMethod == "coherent")
				method = SpinAPI::TraceSamplingMethod::SpinCoherent;

			if (!space.BuildTraceSamples(system->InitialState().front(), plan.monteCarloSamples,
				method, generator, state.traceSamples, &error))
				return false;

			state.factors = state.traceSamples.factors;
			state.factors /= std::sqrt(static_cast<double>(plan.monteCarloSamples));
			// Do not materialize rho = B B^dagger here. Trace sampling exists to
			// keep large nuclear-spin calculations at O(N M), not O(N^2), where
			// M is the number of Monte-Carlo factors. Density construction is
			// reserved for algorithms that intrinsically require it (timeinf or
			// dissipative density propagation; the latter is rejected for stochastic HS).
			state.density.reset();
			state.stochastic = true;

			log << "HSGeneral trace sampling keeps state \"" << system->InitialState().front()->Name()
				<< "\" fixed and samples only omitted spins (subspace dimension "
				<< state.traceSamples.sampledSubspaceDimension << ")." << std::endl;
			log << "Using " << state.factors.n_cols
				<< " normalized trace samples in the shared HSGeneral propagation engine." << std::endl;
		}
		else
		{
			if (state.orientationSpecificThermal)
			{
				log << "HSGeneral will construct the Thermal initial density in the "
					<< "orientation-specific eigen frame of thermalhamiltonian." << std::endl;
			}
			else
			{
				if (!BuildInitialDensity(system, space, state.density, error))
					return false;
				if (!space.FactorizeDensityMatrix(state.density, state.factors, &error))
					return false;
				log << "HSGeneral constructed the normalized initial density from "
					<< system->InitialState().size() << " state component(s) as "
					<< state.factors.n_cols << " Hilbert-space factor(s)." << std::endl;
			}
		}

		state.dephaseInHamiltonianEigenbasis =
			system->InitialStateCoherences() == SpinAPI::InitialStateCoherenceMode::DephaseEigenbasis;
		state.dephasingHamiltonian = plan.hasInitialStateHamiltonian
			? plan.initialStateHamiltonian : plan.h0List;
		if (state.dephaseInHamiltonianEigenbasis && state.dephasingHamiltonian.empty())
		{
			error = "initial-state eigenbasis dephasing requires initialstatehamiltonian or hamiltonianh0list";
			return false;
		}

		if (state.frame == SpinAPI::StateFrame::Molecular && plan.IsPowder())
		{
			// Direct calculations rotate a density matrix; stochastic calculations
			// rotate factors only when the represented density is genuinely
			// orientation dependent. Check stochastic State support sparsely before
			// deciding whether the dense fallback rotation cache is necessary.
			if (state.stochastic)
			{
				arma::sp_cx_mat support;
				if (!space.GetState(system->InitialState().front(), support) ||
					!space.IsStateRotationInvariant(support, state.rotationInvariant))
				{
					error = "failed to determine molecular-frame trace-sample rotation symmetry";
					return false;
				}

				if (!state.rotationInvariant)
				{
					if (!space.CreateStateRotationCache(arma::cx_mat(support), state.rotationCache))
					{
						error = "failed to prepare molecular-frame trace-sample rotations";
						return false;
					}
					state.hasRotationCache = true;
				}
			}
			else if (!space.CreateStateRotationCache(state.density, state.rotationCache))
			{
				error = "failed to prepare molecular-frame initial-state rotations";
				return false;
			}
			else
			{
				state.rotationInvariant = state.rotationCache.rotationInvariant;
				state.hasRotationCache = true;
			}
		}
		return true;
	}

	bool HSStatePreparation::PrepareForOrientation(const HSExecutionPlan &plan,
		SpinAPI::SpinSpace &space, const HSPreparedState &reference,
		const HSOrientation &orientation, HSOrientedState &state, std::string &error)
	{
		state = HSOrientedState();
		error.clear();
		const SpinAPI::HilbertStateRotationCache *rotationCache =
			reference.hasRotationCache ? &reference.rotationCache : nullptr;

		if (reference.stochastic)
		{
			state.factors = reference.factors;
			// For a rotationally invariant trace-sampled density (e.g. singlet
			// electrons times an omitted-spin identity), the random factors are a
			// stochastic quadrature, not physical ensemble members. Reusing the same
			// samples for every crystallite is an unbiased common-random-number
			// estimator and avoids constructing any N x N rotation operator.
			if (reference.frame == SpinAPI::StateFrame::Molecular &&
				!reference.rotationInvariant && reference.hasRotationCache)
			{
				if (!space.RotateStateFactors(reference.factors, orientation.frameToLab,
					reference.rotationCache, state.factors))
				{
					error = "failed to rotate initial trace-sampling factors for the current orientation";
					return false;
				}
			}
			// Keep stochastic propagation factorized. A dense density matrix is
			// intentionally not formed for ordinary finite-time propagation.
			state.density.reset();
			return true;
		}

		arma::cx_mat referenceDensity = reference.density;
		SpinAPI::StateFrame preparationFrame = reference.frame;
		if (reference.orientationSpecificThermal)
		{
			// The canonical density is rho_eq = exp(-hbar H_th/k_B T)/Z.
			// Build H_th with the full molecular-to-laboratory tensor rotation;
			// high-field/secular propagation is a separate approximation and must
			// not silently redefine thermal equilibrium.
			arma::sp_cx_mat thermalHamiltonian;
			if (!space.BaseHamiltonianRotatedZYZ(reference.thermalHamiltonian,
				orientation.frameToLab, thermalHamiltonian) ||
				!space.ThermalStateFromHamiltonian(arma::cx_mat(thermalHamiltonian),
					reference.thermalTemperature, referenceDensity))
			{
				error = "failed to construct the orientation-specific Thermal initial density";
				return false;
			}
			preparationFrame = SpinAPI::StateFrame::Fixed;
		}

		if (!space.PrepareInitialDensityForPowder(referenceDensity, orientation.frameToLab,
			preparationFrame, reference.dephaseInHamiltonianEigenbasis,
			reference.dephasingHamiltonian, plan.approximation, rotationCache, state.density))
		{
			error = "failed to prepare the initial density for the current HS orientation/eigenbasis";
			return false;
		}
		if (!space.FactorizeDensityMatrix(state.density, state.factors, &error))
		{
			error = "failed to factorize the oriented initial density: " + error;
			return false;
		}
		return true;
	}
}
