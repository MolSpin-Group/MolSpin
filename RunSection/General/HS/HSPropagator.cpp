/////////////////////////////////////////////////////////////////////////
// HSPropagator implementation (RunSection::General::HS)
// ------------------
// Hilbert-factor propagation uses dB/dt = (-iH-K)B. Density propagation uses
// drho/dt = -i[H,rho] - {K,rho} + R[rho]. General dissipators therefore select
// density propagation, with exponential coherent/reaction half-steps around the
// dissipative finite step unless propagationmethod=rk4 is requested explicitly.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "HSPropagator.h"
#include "HSReactionRelaxation.h"
#include "SpinSystem.h"
#include "Pulse.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>

namespace RunSection::General::HS
{
	bool HSPropagator::Step(const arma::sp_cx_mat &hamiltonian, const arma::sp_cx_mat &reaction,
		double dt, arma::cx_mat &factors, std::string &error)
	{
		error.clear();
		if (factors.is_empty()) { error = "cannot propagate an empty Hilbert factor"; return false; }
		arma::sp_cx_mat generator = -arma::cx_double(0.0, 1.0) * hamiltonian - reaction;

		switch (plan.propagation)
		{
		case PropagationMethod::AutoExpm:
			factors = space.HighamProp(generator, factors, dt, plan.precision, highamWorkspace);
			return !factors.is_empty();

		case PropagationMethod::Krylov:
		{
			arma::cx_mat next(factors.n_rows, factors.n_cols, arma::fill::zeros);
			for (arma::uword col = 0; col < factors.n_cols; ++col)
			{
				auto result = space.KrylovExpmGeneral(generator, factors.col(col),
					arma::cx_double(dt, 0.0), plan.krylovSize,
					static_cast<int>(factors.n_rows), false, nullptr);
				if (result.result.n_elem != factors.n_rows)
				{ error = "Krylov propagation returned an invalid Hilbert vector"; return false; }
				next.col(col) = result.result;
			}
			factors = std::move(next);
			return true;
		}

		case PropagationMethod::RK4:
		{
			// For a factor B, dB/dt = (-iH-K)B.  RK4 here is intentionally
			// local to the propagation strategy and does not duplicate task logic.
			arma::cx_mat k1 = generator * factors;
			arma::cx_mat k2 = generator * (factors + 0.5 * dt * k1);
			arma::cx_mat k3 = generator * (factors + 0.5 * dt * k2);
			arma::cx_mat k4 = generator * (factors + dt * k3);
			factors += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
			return true;
		}

		case PropagationMethod::Exponential:
		default:
		{
			arma::cx_mat dense = arma::cx_mat(generator);
			arma::cx_mat U = arma::expmat(dt * dense);
			if (U.is_empty()) { error = "matrix exponential propagation failed"; return false; }
			factors = U * factors;
			return true;
		}
		}
	}

	bool HSPropagator::StepDynamicRK4(const arma::sp_cx_mat &hamiltonianStart,
		const arma::sp_cx_mat &reactionStart, const arma::sp_cx_mat &hamiltonianMid,
		const arma::sp_cx_mat &reactionMid, const arma::sp_cx_mat &hamiltonianEnd,
		const arma::sp_cx_mat &reactionEnd, double dt, arma::cx_mat &factors,
		std::string &error)
	{
		error.clear();
		if (factors.is_empty())
		{
			error = "cannot propagate an empty Hilbert factor";
			return false;
		}

		// For dB/dt=G(t)B, classical RK4 needs G at t, t+h/2 and
		// t+h. Freezing G at the midpoint is an exponential-midpoint method,
		// not fourth-order Runge-Kutta for a time-dependent Hamiltonian.
		const arma::sp_cx_mat start = -arma::cx_double(0.0, 1.0) * hamiltonianStart - reactionStart;
		const arma::sp_cx_mat middle = -arma::cx_double(0.0, 1.0) * hamiltonianMid - reactionMid;
		const arma::sp_cx_mat end = -arma::cx_double(0.0, 1.0) * hamiltonianEnd - reactionEnd;
		const arma::cx_mat k1 = start * factors;
		const arma::cx_mat k2 = middle * (factors + 0.5 * dt * k1);
		const arma::cx_mat k3 = middle * (factors + 0.5 * dt * k2);
		const arma::cx_mat k4 = end * (factors + dt * k3);
		factors += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
		return factors.is_finite();
	}

	bool HSPropagator::StepDensity(const arma::sp_cx_mat &hamiltonian,
		const arma::sp_cx_mat &reaction, double dt, arma::cx_mat &density,
		const HSReactionRelaxation &relaxation, const HSRelaxationContext &context,
		std::string &error)
	{
		error.clear();
		if (density.is_empty() || density.n_rows != density.n_cols)
		{ error = "cannot propagate an empty or non-square Hilbert density matrix"; return false; }

		const arma::cx_mat H = arma::cx_mat(hamiltonian);
		const arma::cx_mat K = arma::cx_mat(reaction);
		const arma::cx_double imag(0.0, 1.0);
		auto rhs = [&](const arma::cx_mat &rho, arma::cx_mat &out) -> bool
		{
			out = -imag * (H * rho - rho * H) - (K * rho + rho * K);
			arma::cx_mat relax;
			std::string relaxationError;
			if (!relaxation.ApplyRelaxation(context, rho, relax, relaxationError))
			{
				error = relaxationError;
				return false;
			}
			out += relax;
			return true;
		};

		// A general dissipator does not admit a Hilbert-factor evolution. The
		// frozen direct HS tasks use density-matrix RK4; retain that numerical
		// contract regardless of the factor propagator selected for non-relaxing
		// calculations.
		arma::cx_mat k1, k2, k3, k4;
		if (!rhs(density, k1)) return false;
		if (!rhs(density + 0.5 * dt * k1, k2)) return false;
		if (!rhs(density + 0.5 * dt * k2, k3)) return false;
		if (!rhs(density + dt * k3, k4)) return false;
		density += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
		return true;
	}


	bool HSPropagator::StepDensitySplit(const arma::sp_cx_mat &hamiltonian,
		const arma::sp_cx_mat &reaction, double dt, arma::cx_mat &density,
		const HSReactionRelaxation &relaxation, const HSRelaxationContext &context,
		std::string &error)
	{
		error.clear();
		if (density.is_empty() || density.n_rows != density.n_cols)
		{ error = "cannot propagate an empty or non-square Hilbert density matrix"; return false; }

		const arma::cx_mat generator = -arma::cx_double(0.0, 1.0) * arma::cx_mat(hamiltonian)
			- arma::cx_mat(reaction);
		const arma::cx_mat Uhalf = arma::expmat(generator * (0.5 * dt));
		if (Uhalf.is_empty()) { error = "failed to construct relaxation split Hamiltonian half-step"; return false; }

		density = Uhalf * density * Uhalf.t();
		if (!relaxation.ApplyRelaxationFiniteStep(context, dt, density, error)) return false;
		density = Uhalf * density * Uhalf.t();
		return true;
	}

	bool HSPropagator::StepDensityDynamicRK4(const arma::sp_cx_mat &hamiltonianStart,
		const arma::sp_cx_mat &reactionStart, const arma::sp_cx_mat &hamiltonianMid,
		const arma::sp_cx_mat &reactionMid, const arma::sp_cx_mat &hamiltonianEnd,
		const arma::sp_cx_mat &reactionEnd, double dt, arma::cx_mat &density,
		const HSReactionRelaxation &relaxation, const HSRelaxationContext &context,
		std::string &error)
	{
		error.clear();
		if (density.is_empty() || density.n_rows != density.n_cols)
		{ error = "cannot propagate an empty or non-square Hilbert density matrix"; return false; }

		const arma::cx_mat Hstart(hamiltonianStart);
		const arma::cx_mat Hmid(hamiltonianMid);
		const arma::cx_mat Hend(hamiltonianEnd);
		const arma::cx_mat Kstart(reactionStart);
		const arma::cx_mat Kmid(reactionMid);
		const arma::cx_mat Kend(reactionEnd);
		const arma::cx_double imag(0.0, 1.0);

		auto rhs = [&](const arma::cx_mat &rho, const arma::cx_mat &H,
			const arma::cx_mat &K, arma::cx_mat &out) -> bool
		{
			out = -imag * (H * rho - rho * H) - (K * rho + rho * K);
			arma::cx_mat relax;
			std::string relaxationError;
			if (!relaxation.ApplyRelaxation(context, rho, relax, relaxationError))
			{
				error = relaxationError;
				return false;
			}
			out += relax;
			return true;
		};

		// Match the established DynamicHS direct-relaxation contract: the
		// time-dependent Hamiltonian and reaction operator are evaluated at
		// t, t+dt/2, t+dt/2, and t+dt for the four RK stages.
		arma::cx_mat k1, k2, k3, k4;
		if (!rhs(density, Hstart, Kstart, k1)) return false;
		if (!rhs(density + 0.5 * dt * k1, Hmid, Kmid, k2)) return false;
		if (!rhs(density + 0.5 * dt * k2, Hmid, Kmid, k3)) return false;
		if (!rhs(density + dt * k3, Hend, Kend, k4)) return false;
		density += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
		return true;
	}


	bool HSPropagator::SolveTimeInfinity(const arma::sp_cx_mat &hamiltonian,
		const arma::sp_cx_mat &reaction, const arma::cx_mat &initialDensity,
		const HSReactionRelaxation &relaxation, const HSRelaxationContext &context,
		arma::cx_mat &integratedDensity, std::string &error)
	{
		error.clear();
		arma::cx_mat relaxationSuperoperator;
		if (!context.Empty() &&
			!relaxation.RelaxationSuperoperator(context, relaxationSuperoperator, error))
			return false;

		return space.SolveHilbertTimeIntegral(hamiltonian, reaction,
			relaxationSuperoperator, initialDensity, integratedDensity, &error);
	}


	bool HSPropagator::ApplyPulsePreparationSequence(
		const std::vector<std::tuple<std::string, double>> &sequence,
		const SpinAPI::system_ptr &system, const HSOrientation &orientation,
		const arma::sp_cx_mat &baseHamiltonian, const arma::sp_cx_mat &baseReaction,
		const HSReactionRelaxation &relaxation, const HSRelaxationContext &context,
		bool densityMode, arma::cx_mat &factors, arma::cx_mat &density,
		double &elapsedTime, const HSPulseTimelineObserver &observer,
		std::ostream &log, std::string &error)
	{
		error.clear();
		elapsedTime = 0.0;
		if (sequence.empty()) return true;
		if (system == nullptr) { error = "cannot apply a pulse sequence to a null spin system"; return false; }
		if (plan.IsDynamic())
		{
			error = "task-level pulse timeline currently requires dynamics=static; use time-dependent Interaction objects for continuously driven dynamic propagation";
			return false;
		}

		auto findPulse = [&](const std::string &name) -> SpinAPI::pulse_ptr
		{
			for (const auto &pulse : system->Pulses())
				if (pulse != nullptr && pulse->Name() == name) return pulse;
			return nullptr;
		};

		auto emit = [&]() -> bool
		{
			if (!observer) return true;
			return observer(elapsedTime, densityMode ? density : factors, densityMode, error);
		};

		auto propagate = [&](const arma::sp_cx_mat &H, double dt) -> bool
		{
			if (densityMode)
			{
				if (plan.propagation == PropagationMethod::RK4)
					return this->StepDensity(H, baseReaction, dt, density, relaxation, context, error);
				return this->StepDensitySplit(H, baseReaction, dt, density, relaxation, context, error);
			}
			return this->Step(H, baseReaction, dt, factors, error);
		};

		bool emittedInitialFinitePulseState = false;
		for (const auto &entry : sequence)
		{
			const std::string &pulseName = std::get<0>(entry);
			const double freeDelay = std::get<1>(entry);
			if (!std::isfinite(freeDelay) || freeDelay < 0.0)
			{ error = "pulse-sequence free-evolution delays must be finite and non-negative"; return false; }
			auto pulse = findPulse(pulseName);
			if (pulse == nullptr) { error = "pulse \"" + pulseName + "\" was not found"; return false; }

			const auto type = pulse->Type();
			if (type == SpinAPI::PulseType::MWPulse || type == SpinAPI::PulseType::ShapedPulse ||
				type == SpinAPI::PulseType::Unspecified)
			{
				error = "pulse \"" + pulseName + "\" uses a pulse type that is not yet represented by the generic Hilbert pulse propagator";
				return false;
			}

			// The legacy Pulse API does not expose a crystallite rotation for an
			// anisotropic finite-field pulse. Keep that case explicit; a dynamic
			// Interaction is the orientation-safe representation of a physical B1.
			if (plan.IsPowder() && type != SpinAPI::PulseType::InstantPulse)
			{
				for (bool ignoreTensor : pulse->IgnoreTensorsList())
				{
					if (!ignoreTensor)
					{
						error = "anisotropic finite-field Pulse \"" + pulseName +
							"\" cannot be powder-rotated by the legacy Pulse API; represent the drive as a time-dependent Interaction object";
						return false;
					}
				}
			}

			arma::sp_cx_mat pulseOperator;
			if (!space.PulseOperatorOnStatevector(pulse, pulseOperator))
			{ error = "failed to construct Hilbert pulse operator for \"" + pulseName + "\""; return false; }

			if (type == SpinAPI::PulseType::InstantPulse)
			{
				const arma::cx_mat U(pulseOperator);
				if (densityMode) density = U * density * U.t();
				else factors = U * factors;
				if (!emit()) return false;
				log << "Applied instant pulse \"" << pulseName << "\"." << std::endl;
			}
			else
			{
				const double pulseDt = pulse->Timestep();
				const double pulseTime = pulse->Pulsetime();
				if (!std::isfinite(pulseDt) || !(pulseDt > 0.0) || !std::isfinite(pulseTime) || pulseTime < 0.0)
				{ error = "pulse \"" + pulseName + "\" has an invalid timestep or duration"; return false; }
				if (!emittedInitialFinitePulseState)
				{
					if (!emit()) return false;
					emittedInitialFinitePulseState = true;
				}
				const double ratio = pulseTime / pulseDt;
				if (!std::isfinite(ratio) ||
					ratio > static_cast<double>(std::numeric_limits<size_t>::max()))
				{
					error = "pulse \"" + pulseName + "\" requires too many propagation intervals";
					return false;
				}
				const double nearest = std::round(ratio);
				const double ratioTolerance = 64.0 * std::numeric_limits<double>::epsilon() *
					std::max(1.0, std::abs(ratio));
				const size_t intervals = pulseTime == 0.0 ? 0 : static_cast<size_t>(
					nearest >= 1.0 && std::abs(ratio - nearest) <= ratioTolerance
						? nearest : std::ceil(ratio));
				double pulseElapsed = 0.0;
				for (size_t n = 1; n <= intervals; ++n)
				{
					const double target = n == intervals
						? pulseTime : std::min(pulseTime, static_cast<double>(n) * pulseDt);
					const double interval = target - pulseElapsed;
					if (!(interval > 0.0))
					{
						error = "pulse \"" + pulseName + "\" produced a non-increasing time grid";
						return false;
					}
					double amplitude = 1.0;
					if (type == SpinAPI::PulseType::LongPulse)
						amplitude = std::cos(pulse->Frequency() * target);
					const arma::sp_cx_mat H = baseHamiltonian + amplitude * pulseOperator;
					if (!propagate(H, interval)) return false;
					pulseElapsed = target;
					elapsedTime += interval;
					if (!emit()) return false;
				}
				log << "Applied finite pulse \"" << pulseName << "\" for " << pulseElapsed << " ns." << std::endl;
			}

			if (freeDelay > 0.0)
			{
				double delayDt = pulse->Timestep();
				if (!std::isfinite(delayDt) || !(delayDt > 0.0)) delayDt = plan.timeStep;
				const unsigned int fullSteps = static_cast<unsigned int>(std::floor(freeDelay / delayDt + 1.0e-12));
				for (unsigned int n = 0; n < fullSteps; ++n)
				{
					if (!propagate(baseHamiltonian, delayDt)) return false;
					elapsedTime += delayDt;
					if (!emit()) return false;
				}
				const double remainder = freeDelay - static_cast<double>(fullSteps) * delayDt;
				if (remainder > 1.0e-12)
				{
					if (!propagate(baseHamiltonian, remainder)) return false;
					elapsedTime += remainder;
					if (!emit()) return false;
				}
				log << "Propagated " << freeDelay << " ns of free evolution after pulse \"" << pulseName << "\"." << std::endl;
			}
		}
		return true;
	}

}
