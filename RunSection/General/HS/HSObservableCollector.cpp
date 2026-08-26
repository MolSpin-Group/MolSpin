/////////////////////////////////////////////////////////////////////////
// HSObservableCollector implementation (RunSection::General::HS)
// ------------------
// Observable construction and post-processing for State populations, spin
// polarization, product yields, and CIDSP/CIDNP product polarization.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "HSObservableCollector.h"
#include "../GeneralStateFrame.h"
#include "SpinSystem.h"
#include "State.h"
#include "Transition.h"
#include "Spin.h"

#include <algorithm>
#include <cmath>

namespace RunSection::General::HS
{
	namespace
	{
		bool NameRequested(const std::vector<std::string> &names, const std::string &name)
		{
			return std::find(names.begin(), names.end(), name) != names.end();
		}

		bool PrepareStateRotation(const arma::sp_cx_mat &state, SpinAPI::SpinSpace &space,
			SpinAPI::HilbertStateRotationCache &cache, bool &rotate, std::string &error)
		{
			bool invariant = false;
			if (!space.IsStateRotationInvariant(state, invariant))
			{
				error = "failed to determine powder-rotation symmetry for an HS State/source observable";
				return false;
			}
			if (invariant)
			{
				rotate = false;
				return true;
			}

			if (!space.CreateStateRotationCache(arma::cx_mat(state), cache))
			{
				error = "failed to prepare powder rotation for an orientation-dependent HS State/source observable";
				return false;
			}
			rotate = true;
			return true;
		}
	}

	bool HSObservableCollector::Prepare(const HSExecutionPlan &plan, const SpinAPI::system_ptr &system,
		SpinAPI::SpinSpace &space, std::ostream &log, std::string &error)
	{
		observables.clear();
		error.clear();
		if (system == nullptr) { error = "cannot construct observables for a null spin system"; return false; }

		// Preserve the established DirectSpectra observable contract in auto
		// mode, while allowing General callers to make the output explicit:
		//   transitionyields=true -> rate * P_source
		//   spinlist + cidsp=false -> Ix/Iy/Iz
		//   spinlist + cidsp=true  -> rate * I_alpha * P_source
		// Otherwise time evolution defaults to configured State populations.
		const bool transitionYields =
			plan.observableMode == ObservableMode::TransitionYields ||
			(plan.observableMode == ObservableMode::Auto &&
				plan.calculation == Calculation::Yields && plan.transitionYields);
		const bool productPolarization =
			plan.observableMode == ObservableMode::ProductPolarization ||
			(plan.observableMode == ObservableMode::Auto && plan.cidsp && !plan.spinList.empty());
		const bool spinPolarization =
			plan.observableMode == ObservableMode::SpinPolarization || productPolarization ||
			(plan.observableMode == ObservableMode::Auto && !plan.spinList.empty());
		const bool statePopulations =
			plan.observableMode == ObservableMode::StatePopulations ||
			(plan.observableMode == ObservableMode::Auto &&
				plan.calculation == Calculation::TimeEvolution && plan.spinList.empty());

		if (transitionYields)
		{
			for (const auto &transition : system->Transitions())
			{
				if (transition->SourceState() == nullptr) continue;
				HSObservable observable;
				observable.name = system->Name() + "." + transition->Name() + ".yield";
				if (!space.GetState(transition->SourceState(), observable.stateOrSource))
				{ error = "failed to obtain transition source State \"" + transition->SourceState()->Name() + "\""; return false; }
				observable.hasStateOrSource = true;
				if (plan.IsPowder() &&
					::RunSection::General::TransitionSourceStateFrame(system, transition) == SpinAPI::StateFrame::Molecular)
				{
					if (!PrepareStateRotation(observable.stateOrSource, space,
						observable.stateRotationCache, observable.rotateStateOrSource, error)) return false;
				}
				observable.transition = transition;
				observables.push_back(std::move(observable));
			}
			if (observables.empty()) { error = "transition-yield output requires at least one transition with a source State"; return false; }
			log << "Transition-yield observables selected for " << observables.size()
				<< " reaction channel(s)." << std::endl;
			return true;
		}

		if (spinPolarization)
		{
			std::vector<std::string> found;
			for (auto spin = system->spins_cbegin(); spin != system->spins_cend(); ++spin)
			{
				if (!NameRequested(plan.spinList, (*spin)->Name())) continue;
				found.push_back((*spin)->Name());

				arma::sp_cx_mat Ix, Iy, Iz;
				if (!space.CreateOperator((*spin)->Sx(), *spin, Ix) ||
					!space.CreateOperator((*spin)->Sy(), *spin, Iy) ||
					!space.CreateOperator((*spin)->Sz(), *spin, Iz))
				{
					error = "failed to construct spin-polarization operators for spin \"" + (*spin)->Name() + "\"";
					return false;
				}

				auto appendPolarization = [&](const arma::sp_cx_mat &I, const char *axis,
					const SpinAPI::transition_ptr &transition) -> bool
				{
					HSObservable observable;
					observable.fixedSpinOperator = I;
					observable.hasFixedSpinOperator = true;
					if (transition == nullptr)
					{
						observable.name = system->Name() + "." + (*spin)->Name() + "." + axis;
					}
					else
					{
						if (transition->SourceState() == nullptr) return true;
						observable.name = system->Name() + "." + (*spin)->Name() + "." +
							transition->Name() + ".yield." + axis;
						if (!space.GetState(transition->SourceState(), observable.stateOrSource))
						{ error = "failed to construct CIDSP source State for transition \"" + transition->Name() + "\""; return false; }
						observable.hasStateOrSource = true;
						if (plan.IsPowder() &&
							::RunSection::General::TransitionSourceStateFrame(system, transition) == SpinAPI::StateFrame::Molecular)
						{
							if (!PrepareStateRotation(observable.stateOrSource, space,
								observable.stateRotationCache, observable.rotateStateOrSource, error)) return false;
						}
						observable.transition = transition;
					}
					observables.push_back(std::move(observable));
					return true;
				};

				if (productPolarization)
				{
					for (const auto &transition : system->Transitions())
					{
						if (transition->SourceState() == nullptr) continue;
						if (!appendPolarization(Ix, "Ix", transition) ||
							!appendPolarization(Iy, "Iy", transition) ||
							!appendPolarization(Iz, "Iz", transition)) return false;
					}
				}
				else
				{
					if (!appendPolarization(Ix, "Ix", nullptr) ||
						!appendPolarization(Iy, "Iy", nullptr) ||
						!appendPolarization(Iz, "Iz", nullptr)) return false;
				}
			}

			for (const auto &requested : plan.spinList)
				if (std::find(found.begin(), found.end(), requested) == found.end())
				{ error = "spinlist contains unknown spin \"" + requested + "\""; return false; }
			if (observables.empty()) { error = "spinlist did not produce any polarization observables"; return false; }
			log << (productPolarization ? "CIDSP/CIDNP product-polarization" : "spin-polarization")
				<< " observables selected for " << found.size() << " spin(s)." << std::endl;
			return true;
		}

		if (statePopulations)
		{
			bool hasMolecularFrameObservable = false;
			for (const auto &state : system->States())
			{
				HSObservable observable;
				observable.name = system->Name() + "." + state->Name();
				if (!space.GetState(state, observable.stateOrSource))
				{ error = "failed to obtain State projector \"" + state->Name() + "\""; return false; }
				observable.hasStateOrSource = true;
				const bool molecular =
					::RunSection::General::ObservableStateFrame(system, state) == SpinAPI::StateFrame::Molecular;
				hasMolecularFrameObservable = hasMolecularFrameObservable || molecular;
				if (plan.IsPowder() && molecular)
				{
					if (!PrepareStateRotation(observable.stateOrSource, space,
						observable.stateRotationCache, observable.rotateStateOrSource, error)) return false;
				}
				observables.push_back(std::move(observable));
			}
			if (plan.IsPowder() && hasMolecularFrameObservable)
				log << "Molecular-frame State observables follow each General orientation; fixed/laboratory State observables remain unchanged." << std::endl;
			if (observables.empty()) { error = "time-evolution output requires at least one State or a spinlist"; return false; }
			log << "State-population observables selected for " << observables.size()
				<< " configured State object(s)." << std::endl;
			return true;
		}

		error = "calculation=yields with transitionyields=false requires spinlist";
		return false;
	}

	bool HSObservableCollector::OperatorsForOrientation(SpinAPI::SpinSpace &space,
		const HSOrientation &orientation, std::vector<arma::sp_cx_mat> &operators, std::string &error) const
	{
		operators.clear(); error.clear(); operators.reserve(observables.size());
		for (const auto &observable : observables)
		{
			arma::sp_cx_mat source;
			if (observable.hasStateOrSource)
			{
				source = observable.stateOrSource;
				if (observable.rotateStateOrSource)
				{
					// Spatial rotation currently uses the dense State rotation primitive,
					// but only for genuinely orientation-dependent powder observables.
					// Ordinary/non-powder stochastic calculations retain sparse projectors.
					arma::cx_mat rotated;
					if (!space.RotateState(arma::cx_mat(source), orientation.frameToLab,
						observable.stateRotationCache, rotated))
					{ error = "failed to rotate an HS State/source observable into the current orientation"; return false; }
					source = arma::sp_cx_mat(rotated);
				}
			}

			arma::sp_cx_mat op;
			if (observable.hasFixedSpinOperator && observable.hasStateOrSource)
				op = observable.fixedSpinOperator * source;
			else if (observable.hasFixedSpinOperator)
				op = observable.fixedSpinOperator;
			else
				op = std::move(source);
			operators.push_back(std::move(op));
		}
		return true;
	}

	void HSObservableCollector::Evaluate(const std::vector<arma::sp_cx_mat> &operators,
		const arma::cx_mat &factors, arma::rowvec &values) const
	{
		values.zeros(operators.size());
		const arma::cx_mat conjugate = arma::conj(factors);
		for (size_t i = 0; i < operators.size(); ++i)
		{
			double value = std::real(arma::accu(conjugate % (operators[i] * factors)));
			if (i < observables.size() && observables[i].transition != nullptr)
				value *= observables[i].transition->Rate();
			values(i) = value;
		}
	}

	void HSObservableCollector::EvaluateDensity(const std::vector<arma::sp_cx_mat> &operators,
		const arma::cx_mat &density, arma::rowvec &values) const
	{
		values.zeros(operators.size());
		for (size_t i = 0; i < operators.size(); ++i)
		{
			double value = std::real(arma::trace(operators[i] * density));
			if (i < observables.size() && observables[i].transition != nullptr)
				value *= observables[i].transition->Rate();
			values(i) = value;
		}
	}


	bool HSObservableCollector::IntegrateFiniteTime(const arma::mat &values, double timeStep,
		arma::rowvec &integrated, std::string &error) const
	{
		error.clear();
		integrated.zeros(values.n_cols);
		if (!std::isfinite(timeStep) || !(timeStep > 0.0))
		{
			error = "finite-time observable integration requires a positive finite timestep";
			return false;
		}
		for (arma::uword row = 1; row < values.n_rows; ++row)
			integrated += (0.5 * timeStep) * (values.row(row - 1) + values.row(row));
		return true;
	}

	bool HSObservableCollector::IntegrateTimeline(const std::vector<double> &times,
		const arma::mat &values, bool enabled, arma::mat &output, std::string &error) const
	{
		error.clear();
		output = values;
		if (!enabled || values.n_rows == 0)
			return true;
		if (times.size() != values.n_rows)
		{
			error = "timeline sample times and observable rows have different sizes";
			return false;
		}

		output.zeros(values.n_rows, values.n_cols);
		// Historical DirectSpectra compatibility: the trapezoidal recurrence is
		// accumulated from a zero row 0 and only afterwards is row 0 replaced by
		// the instantaneous initial value. Consequently row 1 contains only the
		// first interval integral rather than initial-value + first interval.
		for (arma::uword row = 1; row < values.n_rows; ++row)
		{
			const double dt = times[row] - times[row - 1];
			if (!std::isfinite(dt) || dt < 0.0)
			{
				error = "timeline sample times must be finite and non-decreasing";
				return false;
			}
			output.row(row) = output.row(row - 1) +
				0.5 * dt * (values.row(row - 1) + values.row(row));
		}
		output.row(0) = values.row(0);
		return true;
	}

	bool HSObservableCollector::ApplyFiniteTimeYieldCorrection(const HSExecutionPlan &plan,
		const SpinAPI::system_ptr &system, SpinAPI::SpinSpace &space,
		arma::rowvec &integrated, std::ostream &log, std::string &error) const
	{
		error.clear();
		if (!plan.yieldCorrections || plan.UsesTimeInfinityYields())
			return true;
		if (space.HasTimedependentTransitions())
		{
			log << "Yield correction is not applied to time-dependent transition rates, matching the legacy dynamic-HS contract." << std::endl;
			return true;
		}
		if (system == nullptr)
		{
			error = "cannot apply a finite-time yield correction to a null spin system";
			return false;
		}

		double maximumRate = 0.0;
		for (const auto &transition : system->Transitions())
			if (transition != nullptr && transition->SourceState() != nullptr)
				maximumRate = std::max(maximumRate, transition->Rate());
		const double denominator = 1.0 - std::exp(-plan.totalTime * maximumRate);
		if (!(denominator > 0.0) || !std::isfinite(denominator))
		{
			error = "yieldcorrections=true requires a positive finite total time and at least one positive transition rate";
			return false;
		}
		integrated /= denominator;
		log << "Applied the legacy finite-time yield correction 1/(1-exp(-totaltime*kmax))." << std::endl;
		return true;
	}

	void HSObservableCollector::WriteHeader(const SpinAPI::system_ptr &, std::ostream &stream) const
	{
		for (const auto &observable : observables) stream << observable.name << " ";
	}
}
