/////////////////////////////////////////////////////////////////////////
// DEVELOPER WORKFLOW / OWNERSHIP MAP
// ----------------------------------------------------------------------
// HSGeneral reaction-loss and Hilbert-relaxation policy.
//
// What is done here:
//   - Builds orientation-aware Haberkorn source-loss operators for Transitions.
//   - Prepares explicit SpinAPI relaxation Operators and phenomenological population/coherence relaxation.
//   - Provides finite-step dissipative application and relaxation-superoperator conversion where needed.
//
// Connections to the General framework / SpinAPI:
//   - Reaction and relaxation objects originate in SpinAPI; this file decides how HSGeneral may use them.
//   - SSGeneral instead builds Liouville-space reaction/relaxation terms; MultiSS owns inter-system transfer edges separately from local relaxation.
//
// Why this ownership is used:
//   - Haberkorn loss is kept in Hilbert form in HSGeneral because the HS propagator can evolve amplitudes/factors directly.
//   - NZ/Redfield are currently superspace theories in the General framework and are therefore not silently approximated by HS relaxation.
//
// Mathematical / physical references:
//   - Haberkorn radical-pair reaction operator; Mol. Phys. 32, 1491-1493 (1976), DOI: 10.1080/00268977600102851.
//
// TODO:
//   - If NZ/Redfield are ever exposed to HSGeneral, reuse one SpinAPI/SS relaxation kernel or a shared generator abstraction; do not duplicate the tensor/correlation algebra here.
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// HSReactionRelaxation implementation (RunSection::General::HS)
// ------------------
// Haberkorn channel selection plus Hilbert relaxation policy. The physical
// k/2 R P R^dagger sink operator and relaxation superoperators are constructed
// by SpinAPI so General/HS does not duplicate spin-space conventions.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "HSReactionRelaxation.h"
#include "../GeneralStateFrame.h"
#include "SpinSystem.h"
#include "Operator.h"
#include "Transition.h"

#include <cmath>

namespace RunSection::General::HS
{
	namespace
	{
		bool OperatorHasNonzeroRate(const SpinAPI::operator_ptr &op)
		{
			return op != nullptr && (op->Rate1() != 0.0 || op->Rate2() != 0.0 || op->Rate3() != 0.0);
		}
	}

	bool HSReactionRelaxation::PrepareReactionChannels(std::string &error)
	{
		error.clear();
		reactionChannels.clear();
		if (system == nullptr) { error = "cannot prepare reactions for a null spin system"; return false; }

		for (const auto &transition : system->Transitions())
		{
			if (transition == nullptr || !transition->IsValid() || transition->SourceState() == nullptr)
			{
				error = "encountered an invalid transition or transition without a source State";
				return false;
			}

			ReactionChannel channel;
			channel.isStatic = SpinAPI::IsStatic(*transition);
			const SpinAPI::StateFrame sourceFrame =
				::RunSection::General::TransitionSourceStateFrame(system, transition);
			if (!::RunSection::General::ValidateProjectorStateFrame(sourceFrame,
				"transition source State \"" + transition->Name() + "\"", error))
				return false;
			channel.rotateSource = sourceFrame == SpinAPI::StateFrame::Molecular;
			if (!space.CreateHilbertReactionOperatorCache(transition, channel.operatorCache,
				plan.IsPowder() && channel.rotateSource))
			{
				error = "failed to prepare the Hilbert reaction operator for transition \"" + transition->Name() + "\"";
				return false;
			}
			reactionChannels.push_back(std::move(channel));
		}
		reactionChannelsPrepared = true;
		return true;
	}

	bool HSReactionRelaxation::Validate(std::string &error)
	{
		error.clear();
		if (system == nullptr) { error = "cannot prepare reactions for a null spin system"; return false; }
		if (plan.IsStochastic() && !system->Operators().empty())
		{
			error = "general Hilbert-space relaxation acts on density matrices and cannot be combined with pure-state trace sampling; use sampling=direct";
			return false;
		}

		if (!PrepareReactionChannels(error)) return false;

		for (const auto &op : system->Operators())
		{
			if (op == nullptr || !op->IsValid())
			{
				error = "encountered an invalid Hilbert-space relaxation operator";
				return false;
			}
			if (op->Type() == SpinAPI::OperatorType::RelaxationPhenomenological)
			{
				if (!std::isfinite(op->Rate1()) || !std::isfinite(op->Rate2()) ||
					op->Rate1() < 0.0 || op->Rate2() < 0.0)
				{
					error = "phenomenological relaxation rates must be finite and non-negative";
					return false;
				}
				continue;
			}

			SpinAPI::HilbertRelaxationCache cache;
			if (!space.RelaxationOperator(op, cache) && OperatorHasNonzeroRate(op))
			{
				error = "Hilbert-space relaxation operator \"" + op->Name() + "\" is not supported by SpinSpace";
				return false;
			}
		}
		return true;
	}

	bool HSReactionRelaxation::BuildReactionForOrientation(const HSOrientation &orientation,
		bool staticPart, bool dynamicPart, arma::sp_cx_mat &reaction, std::string &error) const
	{
		error.clear();
		if (!reactionChannelsPrepared)
		{
			error = "reaction channels were not prepared before propagation";
			return false;
		}
		reaction.zeros(space.HilbertSpaceDimensions(), space.HilbertSpaceDimensions());
		for (const auto &channel : reactionChannels)
		{
			if ((channel.isStatic && !staticPart) || (!channel.isStatic && !dynamicPart))
				continue;
			if (!channel.isStatic && !channel.operatorCache.transition->IsActive())
				continue;

			arma::sp_cx_mat contribution;
			const arma::mat reactionRotation = channel.rotateSource
				? orientation.frameToLab : arma::eye<arma::mat>(3, 3);
			if (!space.ReactionOperatorHilbertRotated(channel.operatorCache,
				reactionRotation, contribution))
			{
				error = "failed to construct the orientation-specific reaction operator for transition \"" +
					channel.operatorCache.transition->Name() + "\"";
				return false;
			}
			reaction += contribution;
		}
		return true;
	}

	bool HSReactionRelaxation::StaticReaction(const HSOrientation &orientation,
		arma::sp_cx_mat &reaction, std::string &error) const
	{
		return BuildReactionForOrientation(orientation, true, !plan.IsDynamic(), reaction, error);
	}

	bool HSReactionRelaxation::ReactionAtTime(double time, const HSOrientation &orientation,
		const arma::sp_cx_mat &staticReaction, arma::sp_cx_mat &reaction, std::string &error)
	{
		error.clear();
		if (!plan.IsDynamic() || !space.HasTimedependentTransitions()) { reaction = staticReaction; return true; }
		space.SetTime(time);
		arma::sp_cx_mat dynamic;
		if (!BuildReactionForOrientation(orientation, false, true, dynamic, error)) return false;
		reaction = staticReaction + dynamic;
		return true;
	}

	bool HSReactionRelaxation::HasRelaxation() const
	{
		return system != nullptr && !system->Operators().empty();
	}

	bool HSReactionRelaxation::PrepareRelaxation(const HSOrientation &orientation,
		const arma::sp_cx_mat &basisHamiltonian, HSRelaxationContext &context,
		std::string &error) const
	{
		context = HSRelaxationContext();
		error.clear();
		if (system == nullptr) { error = "cannot prepare relaxation for a null spin system"; return false; }

		for (const auto &op : system->Operators())
		{
			if (op->Type() == SpinAPI::OperatorType::RelaxationPhenomenological)
			{
				if (op->Rate1() == 0.0 && op->Rate2() == 0.0) continue;
				SpinAPI::HilbertRelaxationPhenomenologicalTerm term;
				term.populationRate = op->Rate1();
				term.coherenceRate = op->Rate2();
				context.phenomenologicalTerms.push_back(term);
				context.hasPhenomenological = true;
				continue;
			}

			bool added = plan.IsPowder()
				? space.PowderRelaxationOperatorHilbert(op, orientation.frameToLab, context.explicitCache)
				: space.RelaxationOperator(op, context.explicitCache);
			if (added) context.hasExplicit = true;
			else if (OperatorHasNonzeroRate(op))
			{
				error = "failed to construct Hilbert-space relaxation operator \"" + op->Name() + "\" for the current orientation";
				return false;
			}
		}

		if (context.hasPhenomenological)
		{
			arma::vec eigenvalues;
			if (!arma::eig_sym(eigenvalues, context.phenomenologicalBasis, arma::cx_mat(basisHamiltonian)))
			{
				error = "failed to diagonalize the Hamiltonian defining the phenomenological relaxation basis";
				return false;
			}
		}
		return true;
	}

	bool HSReactionRelaxation::ApplyRelaxation(const HSRelaxationContext &context,
		const arma::cx_mat &rho, arma::cx_mat &out, std::string &error) const
	{
		error.clear();
		out.zeros(rho.n_rows, rho.n_cols);
		if (context.hasExplicit)
		{
			arma::cx_mat explicitTerm;
			if (!space.ApplyRelaxationHilbert(context.explicitCache, rho, explicitTerm))
			{ error = "failed to apply explicit Hilbert-space relaxation"; return false; }
			out += explicitTerm;
		}
		if (context.hasPhenomenological)
		{
			arma::cx_mat phenomenologicalTerm;
			if (!space.ApplyPhenomenologicalRelaxationHilbert(context.phenomenologicalTerms,
				context.phenomenologicalBasis, rho, phenomenologicalTerm))
			{ error = "failed to apply phenomenological Hilbert-space relaxation"; return false; }
			out += phenomenologicalTerm;
		}
		return true;
	}


	bool HSReactionRelaxation::ApplyRelaxationFiniteStep(const HSRelaxationContext &context,
		double dt, arma::cx_mat &rho, std::string &error) const
	{
		error.clear();
		if (context.Empty() || dt == 0.0) return true;
		if (!std::isfinite(dt) || dt < 0.0)
		{
			error = "relaxation finite-step duration must be finite and non-negative";
			return false;
		}

		// Match the mature StaticHS-Direct-Spectra relaxation contract.
		// Pure phenomenological relaxation has an exact finite-step map in the
		// orientation-specific H0 eigenbasis. When explicit spin-operator
		// relaxation is present, integrate only the dissipative part by RK4;
		// the Hamiltonian/reaction evolution is handled by exponential half
		// steps in HSPropagator, avoiding the severe timestep restriction of a
		// full-Hamiltonian RK4 treatment at EPR frequencies.
		if (context.hasPhenomenological && !context.hasExplicit)
		{
			SpinAPI::HilbertPhenomenologicalRelaxationMap map;
			if (!space.CreatePhenomenologicalRelaxationMapHilbert(
				context.phenomenologicalTerms, context.phenomenologicalBasis, dt, map))
			{
				error = "failed to construct the exact phenomenological relaxation map";
				return false;
			}
			arma::cx_mat workspace;
			if (!space.ApplyPhenomenologicalRelaxationMapHilbert(map, rho, workspace))
			{
				error = "failed to apply the exact phenomenological relaxation map";
				return false;
			}
			return true;
		}

		auto rhs = [&](const arma::cx_mat &state, arma::cx_mat &out) -> bool
		{
			return this->ApplyRelaxation(context, state, out, error);
		};
		arma::cx_mat k1, k2, k3, k4;
		if (!rhs(rho, k1)) return false;
		if (!rhs(rho + 0.5 * dt * k1, k2)) return false;
		if (!rhs(rho + 0.5 * dt * k2, k3)) return false;
		if (!rhs(rho + dt * k3, k4)) return false;
		rho += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
		return true;
	}

	bool HSReactionRelaxation::RelaxationSuperoperator(const HSRelaxationContext &context,
		arma::cx_mat &superoperator, std::string &error) const
	{
		error.clear();
		superoperator.reset();
		if (context.hasExplicit)
		{
			if (!space.RelaxationSuperoperatorHilbert(context.explicitCache, superoperator))
			{ error = "failed to construct the explicit Hilbert relaxation superoperator"; return false; }
		}
		if (context.hasPhenomenological)
		{
			arma::cx_mat phenomenologicalSuperoperator;
			if (!space.PhenomenologicalRelaxationSuperoperatorHilbert(context.phenomenologicalTerms,
				context.phenomenologicalBasis, phenomenologicalSuperoperator))
			{ error = "failed to construct the phenomenological Hilbert relaxation superoperator"; return false; }
			if (superoperator.is_empty()) superoperator = std::move(phenomenologicalSuperoperator);
			else superoperator += phenomenologicalSuperoperator;
		}
		return true;
	}
}
