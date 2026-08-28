/////////////////////////////////////////////////////////////////////////
// DEVELOPER WORKFLOW / OWNERSHIP MAP
// ----------------------------------------------------------------------
// Top-level HSGeneral orchestration.
//
// What is done here:
//   - Validates one SpinSystem, resolves the HS execution plan and builds the orientation ensemble.
//   - For every crystallite: prepares state/Hamiltonian/reaction/relaxation, propagates or solves, evaluates observables, and accumulates the weighted result.
//   - Owns output/log sequencing, not low-level spin physics.
//
// Connections to the General framework / SpinAPI:
//   - Delegates to HSExecutionPlan, HSOrientationSampler, HSStatePreparation, HSHamiltonianBuilder, HSReactionRelaxation, HSPropagator and HSObservableCollector.
//   - SpinAPI owns spins, interactions, states, transitions, Hamiltonians, rotations and low-level numerical kernels.
//   - SSGeneral is the one-manifold Liouville analogue; MultiSSGeneral is the kinetic network analogue.
//
// Why this ownership is used:
//   - The task remains an orchestrator so individual numerical/physical layers can be validated independently against legacy tasks.
//
// TODO:
//   - General ensemble extensions (strain/SW) should wrap this per-orientation calculation instead of being embedded in TaskHSGeneral-specific loops.
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// TaskHSGeneral implementation (RunSection::General::HS)
// ------------------
// Production Hilbert-space RunSection task. This class owns RunSection lifecycle,
// logging, output, and orchestration only; physical/numerical operations are
// delegated to the General/HS components and SpinAPI.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "TaskHSGeneral.h"

#include "HSHamiltonianBuilder.h"
#include "HSObservableCollector.h"
#include "HSOrientationSampler.h"
#include "HSPropagator.h"
#include "HSReactionRelaxation.h"
#include "HSStatePreparation.h"
#include "../GeneralLog.h"

#include "ObjectParser.h"
#include "Pulse.h"
#include "Settings.h"
#include "SpinSystem.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <random>

namespace RunSection::General::HS
{
	namespace
	{
		bool BuildFreeEvolutionTimes(double totalTime, double timeStep,
			std::vector<double> &times, std::string &error)
		{
			times.clear();
			times.push_back(0.0);
			if (totalTime == 0.0)
				return true;

			// Build targets from integer indices and set the final target exactly
			// to totalTime.  This avoids both cumulative floating-point drift and
			// the former ceil(total/dt)-row bug that omitted the requested endpoint.
			const double ratio = totalTime / timeStep;
			if (!std::isfinite(ratio) ||
				ratio > static_cast<double>(std::numeric_limits<size_t>::max() - 1))
			{
				error = "HSGeneral free evolution requires too many time intervals";
				return false;
			}
			const double nearest = std::round(ratio);
			const double ratioTolerance = 64.0 * std::numeric_limits<double>::epsilon() *
				std::max(1.0, std::abs(ratio));
			const size_t intervals = static_cast<size_t>(
				nearest >= 1.0 && std::abs(ratio - nearest) <= ratioTolerance
					? nearest : std::ceil(ratio));
			times.reserve(intervals + 1);
			for (size_t step = 1; step <= intervals; ++step)
			{
				const double target = step == intervals
					? totalTime : std::min(totalTime, static_cast<double>(step) * timeStep);
				if (!(target > times.back()))
				{
					error = "HSGeneral produced a non-increasing free-evolution time grid";
					return false;
				}
				times.push_back(target);
			}
			return true;
		}
	}

	TaskHSGeneral::TaskHSGeneral(const MSDParser::ObjectParser &parser, const RunSection &runsection)
		: BasicTask(parser, runsection)
	{
	}

	bool TaskHSGeneral::WriteHeader(std::ostream &stream)
	{
		stream << "Step ";
		if (plan.calculation == Calculation::TimeEvolution)
			stream << "Time(ns) ";
		this->WriteStandardOutputHeader(stream);

		for (const auto &system : this->SpinSystems())
		{
			SpinAPI::SpinSpace space(system);
			space.UseSuperoperatorSpace(false);

			HSObservableCollector collector;
			std::string error;
			if (!collector.Prepare(plan, system, space, this->Log(), error))
			{
				this->Log() << "ERROR: failed to construct HSGeneral output header: "
					<< error << "." << std::endl;
				return false;
			}
			collector.WriteHeader(system, stream);
		}
		stream << std::endl;
		return true;
	}

	bool TaskHSGeneral::Validate()
	{
		std::string error;
		if (!ResolveExecutionPlan(*this->Properties(), plan, error))
		{
			this->Log() << "ERROR: Invalid HSGeneral execution plan: " << error << "." << std::endl;
			return false;
		}
		planResolved = true;

		const auto systems = this->SpinSystems();
		if (systems.size() != 1)
		{
			this->Log() << "ERROR: TaskHSGeneral owns one SpinSystem per task. "
				<< "Multi-system propagation belongs to MultiSpinSystemGeneral; use an existing "
				<< "multi-system task until that framework is consolidated." << std::endl;
			return false;
		}

		if (plan.krylovToleranceSpecified)
		{
			this->Log() << "Warning: krylovtol = " << plan.requestedKrylovTolerance
				<< " is parsed for compatibility but is not used by SpinAPI::KrylovExpmGeneral; "
				<< "krylovsize controls the current Krylov branch." << std::endl;
		}
		if (plan.IsDynamic() && plan.orientation == OrientationMode::Powder2D)
		{
			this->Log() << "Warning: dynamic theta/phi powder sampling assumes axial symmetry about the laboratory field. "
				<< "A linearly polarized drive or any second fixed laboratory direction requires full SO(3) sampling "
				<< "with powdergammapoints greater than one." << std::endl;
		}
		if (plan.orientation == OrientationMode::PowderSO3 &&
			plan.powderGridType == SpinAPI::PowderGridType::Uniform &&
			plan.powderDomain == SpinAPI::PowderGridDomain::UpperHemisphere)
		{
			this->Log() << "Warning: powderdomain=upper with gamma sampling is symmetry-reduced, not a generic full SO(3) integral. "
				<< "Use it only when inversion symmetry has been established; otherwise select powderdomain=full." << std::endl;
		}

		for (const auto &system : systems)
		{
			if (system == nullptr)
				return false;

			SpinAPI::SpinSpace space(system);
			space.UseSuperoperatorSpace(false);

			if (plan.hasPulseSequence)
			{
				if (plan.IsDynamic())
				{
					this->Log() << "ERROR: task-level pulse timeline currently requires dynamics=static. "
						<< "Use time-dependent Interaction objects for continuously driven propagation."
						<< std::endl;
					return false;
				}

				const auto pulses = system->Pulses();
				for (const auto &entry : plan.pulseSequence)
				{
					const std::string &name = std::get<0>(entry);
					const auto found = std::find_if(pulses.begin(), pulses.end(),
						[&](const SpinAPI::pulse_ptr &pulse)
						{ return pulse != nullptr && pulse->Name() == name; });
					if (found == pulses.end())
					{
						this->Log() << "ERROR: pulsesequence references unknown Pulse \""
							<< name << "\"." << std::endl;
						return false;
					}
				}
			}

			if (plan.IsDynamic() && !space.HasTimedependentInteractions() &&
				!space.HasTimedependentTransitions())
			{
				this->Log() << "ERROR: dynamics=dynamic requires a time-dependent interaction or transition."
					<< std::endl;
				return false;
			}
			if (plan.IsStochastic() && !HSStatePreparation::ValidateTraceSampling(system, error))
			{
				this->Log() << "ERROR: " << error << "." << std::endl;
				return false;
			}

			HSReactionRelaxation reaction(plan, system, space);
			if (!reaction.Validate(error))
			{
				this->Log() << "ERROR: " << error << "." << std::endl;
				return false;
			}
			if (reaction.HasRelaxation() && plan.propagation != PropagationMethod::RK4)
			{
				this->Log() << "Relaxation operators detected: HSGeneral will use density-matrix "
					<< "propagation with exponential Hamiltonian/reaction splitting by default; "
					<< "propagationmethod=rk4 explicitly requests full density RK4." << std::endl;
			}

			HSObservableCollector observables;
			if (!observables.Prepare(plan, system, space, this->Log(), error))
			{
				this->Log() << "ERROR: " << error << "." << std::endl;
				return false;
			}
		}

		this->Log() << "\n--- HSGeneral resolved calculation ---" << std::endl;
		this->Log() << "Task mode: dynamics=" << ToString(plan.dynamics)
			<< ", calculation=" << ToString(plan.calculation)
			<< ", sampling=" << ToString(plan.sampling)
			<< ", orientation=" << ToString(plan.orientation)
			<< ", observables=" << ToString(plan.observableMode)
			<< ", propagation=" << ToString(plan.propagation);
		if (plan.calculation == Calculation::Yields)
		{
			this->Log() << ", yield-mode=" << ToString(plan.yieldMode)
				<< ", yield-correction=" << (plan.yieldCorrections ? "on" : "off");
		}
		this->Log() << "." << std::endl;

		this->Log() << "Hamiltonian approximation: "
			<< (plan.approximation == SpinAPI::HamiltonianApproximation::Full
				? "full" : "high-field secular") << "." << std::endl;
		if (plan.approximation != SpinAPI::HamiltonianApproximation::Full)
		{
			this->Log() << "    Note: static high-field secularization is not a rotating-wave approximation. "
				<< "RWA/frequency/selective-spin handling remains defined by the relevant time-dependent Interaction/Pulse."
				<< std::endl;
		}
		::RunSection::General::Log::PrintSystemInventory(this->Log(), systems, "HSGeneral physics objects");
		return true;
	}

	bool TaskHSGeneral::RunLocal()
	{
		if (!planResolved)
		{
			std::string error;
			if (!ResolveExecutionPlan(*this->Properties(), plan, error))
				return false;
		}

		this->Log() << "Running unified TaskHSGeneral." << std::endl;
		if (this->RunSettings()->CurrentStep() == 1 && !this->WriteHeader(this->Data()))
			return false;

		std::vector<HSOrientation> orientations;
		std::string error;
		if (!HSOrientationSampler::Build(plan, orientations, this->Log(), error))
		{
			this->Log() << "ERROR: " << error << "." << std::endl;
			return false;
		}
		double orientationWeightSum = 0.0;
		for (const auto &orientation : orientations) orientationWeightSum += orientation.weight;
		::RunSection::General::Log::PrintPowderSummary(this->Log(), ToString(plan.orientation),
			orientations.size(), orientationWeightSum);

		std::random_device randomDevice;
		std::mt19937 generator(randomDevice());
		HSStatePreparation::SeedGenerator(plan, generator, this->Log());
		std::vector<double> freeTimes;
		if (!BuildFreeEvolutionTimes(plan.totalTime, plan.timeStep, freeTimes, error))
		{
			this->Log() << "ERROR: " << error << "." << std::endl;
			return false;
		}
		const size_t numSteps = freeTimes.size();

		for (const auto &system : this->SpinSystems())
		{
			this->Log() << "\nStarting unified HS propagation for SpinSystem \""
				<< system->Name() << "\"." << std::endl;

			SpinAPI::SpinSpace space(system);
			space.UseSuperoperatorSpace(false);

			HSPreparedState prepared;
			if (!HSStatePreparation::Prepare(plan, system, space, prepared,
				generator, this->Log(), error))
			{
				this->Log() << "ERROR: " << error << "." << std::endl;
				return false;
			}

			HSObservableCollector collector;
			if (!collector.Prepare(plan, system, space, this->Log(), error))
			{
				this->Log() << "ERROR: " << error << "." << std::endl;
				return false;
			}

			HSReactionRelaxation reaction(plan, system, space);
			if (!reaction.Validate(error))
			{
				this->Log() << "ERROR: " << error << "." << std::endl;
				return false;
			}
			HSHamiltonianBuilder hamiltonianBuilder(plan, space);
			HSPropagator propagator(plan, space);
			const bool useDensityPropagation = reaction.HasRelaxation();

			arma::mat averagedFree;
			if (plan.calculation == Calculation::TimeEvolution)
				averagedFree.zeros(numSteps, collector.Size());
			arma::mat averagedPulse;
			std::vector<double> pulseTimes;
			double pulseElapsedReference = 0.0;
			bool havePulseTimeline = false;
			arma::rowvec averagedYield(collector.Size(), arma::fill::zeros);

			for (size_t orientationIndex = 0; orientationIndex < orientations.size(); ++orientationIndex)
			{
				const auto &orientation = orientations[orientationIndex];
				::RunSection::General::Log::PrintOrientationProgress(this->Log(), orientationIndex, orientations.size());
				HSOrientedState orientedState;
				if (!HSStatePreparation::PrepareForOrientation(plan, space, prepared,
					orientation, orientedState, error))
				{
					this->Log() << "ERROR: " << error << "." << std::endl;
					return false;
				}
				arma::cx_mat factors = orientedState.factors;
				arma::cx_mat density;
				if (useDensityPropagation)
					density = orientedState.density;

				std::vector<arma::sp_cx_mat> operators;
				if (!collector.OperatorsForOrientation(space, orientation, operators, error))
				{
					this->Log() << "ERROR: " << error << "." << std::endl;
					return false;
				}

				arma::sp_cx_mat staticHamiltonian;
				arma::sp_cx_mat h0;
				arma::sp_cx_mat staticReaction;
				if (!hamiltonianBuilder.BuildStatic(orientation, staticHamiltonian, &h0, error) ||
					!reaction.StaticReaction(orientation, staticReaction, error))
				{
					this->Log() << "ERROR: " << error << "." << std::endl;
					return false;
				}

				HSRelaxationContext relaxationContext;
				if (!reaction.PrepareRelaxation(orientation, h0, relaxationContext, error))
				{
					this->Log() << "ERROR: " << error << "." << std::endl;
					return false;
				}

				std::vector<double> orientationPulseTimes;
				std::vector<arma::rowvec> orientationPulseRows;
				double pulseElapsed = 0.0;
				HSPulseTimelineObserver pulseObserver;
				if (plan.calculation == Calculation::TimeEvolution && plan.hasPulseSequence)
				{
					pulseObserver = [&](double time, const arma::cx_mat &state,
						bool isDensity, std::string &) -> bool
					{
						arma::rowvec values;
						if (isDensity)
							collector.EvaluateDensity(operators, state, values);
						else
							collector.Evaluate(operators, state, values);
						orientationPulseTimes.push_back(time);
						orientationPulseRows.push_back(std::move(values));
						return true;
					};
				}

				if (plan.hasPulseSequence &&
					!propagator.ApplyPulsePreparationSequence(plan.pulseSequence, system, orientation,
						staticHamiltonian, staticReaction, reaction, relaxationContext,
						useDensityPropagation, factors, density, pulseElapsed,
						pulseObserver, this->Log(), error))
				{
					this->Log() << "ERROR: " << error << "." << std::endl;
					return false;
				}

				if (plan.calculation == Calculation::TimeEvolution && !orientationPulseRows.empty())
				{
					arma::mat pulseMatrix(orientationPulseRows.size(), collector.Size(), arma::fill::zeros);
					for (size_t row = 0; row < orientationPulseRows.size(); ++row)
						pulseMatrix.row(row) = orientationPulseRows[row];

					if (!havePulseTimeline)
					{
						pulseTimes = orientationPulseTimes;
						pulseElapsedReference = pulseElapsed;
						averagedPulse.zeros(pulseMatrix.n_rows, pulseMatrix.n_cols);
						havePulseTimeline = true;
					}
					else
					{
						if (pulseTimes.size() != orientationPulseTimes.size() ||
							std::abs(pulseElapsedReference - pulseElapsed) > 1.0e-10)
						{
							this->Log() << "ERROR: pulse timeline changed between powder orientations."
								<< std::endl;
							return false;
						}
						for (size_t row = 0; row < pulseTimes.size(); ++row)
						{
							if (std::abs(pulseTimes[row] - orientationPulseTimes[row]) > 1.0e-10)
							{
								this->Log() << "ERROR: pulse sample times changed between powder orientations."
									<< std::endl;
								return false;
							}
						}
					}
					averagedPulse += orientation.weight * pulseMatrix;
				}

				if (plan.UsesTimeInfinityYields())
				{
					const arma::cx_mat initialDensity = useDensityPropagation
						? density : factors * factors.t();
					arma::cx_mat integratedDensity;
					if (!propagator.SolveTimeInfinity(staticHamiltonian, staticReaction,
						initialDensity, reaction, relaxationContext, integratedDensity, error))
					{
						this->Log() << "ERROR: " << error << "." << std::endl;
						return false;
					}
					arma::rowvec integratedValues;
					collector.EvaluateDensity(operators, integratedDensity, integratedValues);
					averagedYield += orientation.weight * integratedValues;
					continue;
				}

				arma::mat orientationValues(numSteps, collector.Size(), arma::fill::zeros);
				for (size_t step = 0; step < numSteps; ++step)
				{
					const double currentTime = freeTimes[step];
					if (plan.IsDynamic())
						space.SetTime(currentTime);

					arma::rowvec values;
					if (useDensityPropagation)
						collector.EvaluateDensity(operators, density, values);
					else
						collector.Evaluate(operators, factors, values);
					orientationValues.row(step) = values;
					if (step + 1 >= numSteps)
						continue;

					const double nextTime = freeTimes[step + 1];
					const double interval = nextTime - currentTime;
					const double midpoint = currentTime + 0.5 * interval;
					// Density RK4 already needs the true time-dependent stages.  The
					// same is required for factor propagation when the user explicitly
					// selects RK4.  Exponential factor propagation continues to use the
					// second-order midpoint Hamiltonian below.
					if (plan.IsDynamic() &&
						(useDensityPropagation || plan.propagation == PropagationMethod::RK4))
					{
						arma::sp_cx_mat hamiltonianStart, hamiltonianMid, hamiltonianEnd;
						arma::sp_cx_mat reactionStart, reactionMid, reactionEnd;
						if (!hamiltonianBuilder.BuildAtTime(orientation, currentTime,
							staticHamiltonian, hamiltonianStart, error) ||
							!reaction.ReactionAtTime(currentTime, orientation, staticReaction,
								reactionStart, error) ||
							!hamiltonianBuilder.BuildAtTime(orientation, midpoint,
								staticHamiltonian, hamiltonianMid, error) ||
							!reaction.ReactionAtTime(midpoint, orientation, staticReaction,
								reactionMid, error) ||
							!hamiltonianBuilder.BuildAtTime(orientation, nextTime,
								staticHamiltonian, hamiltonianEnd, error) ||
							!reaction.ReactionAtTime(nextTime, orientation,
								staticReaction, reactionEnd, error))
						{
							this->Log() << "ERROR: " << error << "." << std::endl;
							return false;
						}
						const bool propagated = useDensityPropagation
							? propagator.StepDensityDynamicRK4(hamiltonianStart, reactionStart,
								hamiltonianMid, reactionMid, hamiltonianEnd, reactionEnd,
								interval, density, reaction, relaxationContext, error)
							: propagator.StepDynamicRK4(hamiltonianStart, reactionStart,
								hamiltonianMid, reactionMid, hamiltonianEnd, reactionEnd,
								interval, factors, error);
						if (!propagated)
						{
							this->Log() << "ERROR: " << error << "." << std::endl;
							return false;
						}
					}
					else
					{
						arma::sp_cx_mat hamiltonian;
						arma::sp_cx_mat reactionOperator;
						if (!hamiltonianBuilder.BuildAtTime(orientation, midpoint,
							staticHamiltonian, hamiltonian, error) ||
							!reaction.ReactionAtTime(midpoint, orientation, staticReaction,
								reactionOperator, error))
						{
							this->Log() << "ERROR: " << error << "." << std::endl;
							return false;
						}

						if (useDensityPropagation)
						{
							const bool ok = plan.propagation == PropagationMethod::RK4
								? propagator.StepDensity(hamiltonian, reactionOperator, interval,
									density, reaction, relaxationContext, error)
								: propagator.StepDensitySplit(hamiltonian, reactionOperator, interval,
									density, reaction, relaxationContext, error);
							if (!ok)
							{
								this->Log() << "ERROR: " << error << "." << std::endl;
								return false;
							}
						}
						else if (!propagator.Step(hamiltonian, reactionOperator,
							interval, factors, error))
						{
							this->Log() << "ERROR: " << error << "." << std::endl;
							return false;
						}
					}
				}

				if (plan.calculation == Calculation::TimeEvolution)
				{
					averagedFree += orientation.weight * orientationValues;
				}
				else
				{
					arma::rowvec integratedValues;
					if (!collector.IntegrateFiniteTime(orientationValues, freeTimes,
						integratedValues, error))
					{
						this->Log() << "ERROR: " << error << "." << std::endl;
						return false;
					}
					averagedYield += orientation.weight * integratedValues;
				}
			}

			if (plan.calculation == Calculation::Yields &&
				!collector.ApplyFiniteTimeYieldCorrection(plan, system, space,
					averagedYield, this->Log(), error))
			{
				this->Log() << "ERROR: " << error << "." << std::endl;
				return false;
			}

			if (plan.calculation == Calculation::TimeEvolution)
			{
				const bool integratePulse = plan.integrateTimeEvolution &&
					(plan.integrationWindow == TimelineWindow::Pulse ||
						plan.integrationWindow == TimelineWindow::Full);
				const bool integrateFree = plan.integrateTimeEvolution &&
					(plan.integrationWindow == TimelineWindow::FreeEvolution ||
						plan.integrationWindow == TimelineWindow::Full);

				arma::mat pulseOutput;
				arma::mat freeOutput;
				if (!collector.IntegrateTimeline(pulseTimes, averagedPulse, integratePulse,
					pulseOutput, error) ||
					!collector.IntegrateTimeline(freeTimes, averagedFree, integrateFree,
						freeOutput, error))
				{
					this->Log() << "ERROR: " << error << "." << std::endl;
					return false;
				}

				auto writeTimeline = [&](const std::vector<double> &times,
					const arma::mat &values, double offset)
				{
					for (arma::uword row = 0; row < values.n_rows; ++row)
					{
						this->Data() << this->RunSettings()->CurrentStep() << " "
							<< std::setprecision(12) << offset + times[row] << " ";
						this->WriteStandardOutput(this->Data());
						for (arma::uword col = 0; col < values.n_cols; ++col)
							this->Data() << " " << std::setprecision(12) << values(row, col);
						this->Data() << std::endl;
					}
				};

				if (plan.printWindow == TimelineWindow::Pulse ||
					plan.printWindow == TimelineWindow::Full)
					writeTimeline(pulseTimes, pulseOutput, 0.0);
				if (plan.printWindow == TimelineWindow::FreeEvolution ||
					plan.printWindow == TimelineWindow::Full)
				{
					const double offset = plan.printWindow == TimelineWindow::Full && havePulseTimeline
						? pulseElapsedReference : 0.0;
					writeTimeline(freeTimes, freeOutput, offset);
				}
				if (plan.integrateTimeEvolution)
				{
					this->Log() << "Segment-aware integration enabled for "
						<< ToString(plan.integrationWindow) << " timeline output." << std::endl;
				}
			}
			else
			{
				this->Log() << (plan.UsesTimeInfinityYields()
					? "Writing time-infinity integrated HS observables for the static Hamiltonian."
					: "Writing finite-time integrated HS observables.") << std::endl;
				this->Data() << this->RunSettings()->CurrentStep() << " ";
				this->WriteStandardOutput(this->Data());
				for (arma::uword col = 0; col < averagedYield.n_elem; ++col)
					this->Data() << " " << std::setprecision(12) << averagedYield(col);
				this->Data() << std::endl;
			}

			this->Log() << "Done with SpinSystem \"" << system->Name() << "\"." << std::endl;
		}
		return true;
	}
}
