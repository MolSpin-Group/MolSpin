/////////////////////////////////////////////////////////////////////////
// HSReactionRelaxation (RunSection::General::HS)
// ------------------
// Coordinates Hilbert-space Haberkorn sink loss and relaxation. Reaction-operator
// construction/rotation remains a SpinAPI responsibility; this component selects
// static/dynamic channels and prepares explicit/phenomenological relaxation caches.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_HS_HSReactionRelaxation
#define MOD_RunSection_General_HS_HSReactionRelaxation

#include <armadillo>
#include <string>
#include <vector>
#include "SpinAPIfwd.h"
#include "SpinSpace.h"
#include "HSExecutionPlan.h"
#include "HSOrientationSampler.h"

namespace RunSection::General::HS
{
	struct HSRelaxationContext
	{
		SpinAPI::HilbertRelaxationCache explicitCache;
		std::vector<SpinAPI::HilbertRelaxationPhenomenologicalTerm> phenomenologicalTerms;
		arma::cx_mat phenomenologicalBasis;
		bool hasExplicit = false;
		bool hasPhenomenological = false;

		bool Empty() const { return !hasExplicit && !hasPhenomenological; }
	};

	class HSReactionRelaxation
	{
	public:
		HSReactionRelaxation(const HSExecutionPlan &_plan, const SpinAPI::system_ptr &_system,
			SpinAPI::SpinSpace &_space) : plan(_plan), system(_system), space(_space) {}

		bool Validate(std::string &_error);
		bool StaticReaction(const HSOrientation &_orientation, arma::sp_cx_mat &_reaction,
			std::string &_error) const;
		bool ReactionAtTime(double _time, const HSOrientation &_orientation,
			const arma::sp_cx_mat &_staticReaction, arma::sp_cx_mat &_reaction,
			std::string &_error);

		bool HasRelaxation() const;
		bool PrepareRelaxation(const HSOrientation &_orientation,
			const arma::sp_cx_mat &_basisHamiltonian, HSRelaxationContext &_context,
			std::string &_error) const;
		bool ApplyRelaxation(const HSRelaxationContext &_context, const arma::cx_mat &_rho,
			arma::cx_mat &_out, std::string &_error) const;
		bool ApplyRelaxationFiniteStep(const HSRelaxationContext &_context, double _dt,
			arma::cx_mat &_rho, std::string &_error) const;
		bool RelaxationSuperoperator(const HSRelaxationContext &_context,
			arma::cx_mat &_superoperator, std::string &_error) const;

	private:
		struct ReactionChannel
		{
			SpinAPI::HilbertReactionOperatorCache operatorCache;
			bool isStatic = true;
			bool rotateSource = false;
		};

		bool PrepareReactionChannels(std::string &_error);
		bool BuildReactionForOrientation(const HSOrientation &_orientation, bool _staticPart,
			bool _dynamicPart, arma::sp_cx_mat &_reaction, std::string &_error) const;

		const HSExecutionPlan &plan;
		SpinAPI::system_ptr system;
		SpinAPI::SpinSpace &space;
		std::vector<ReactionChannel> reactionChannels;
		bool reactionChannelsPrepared = false;
	};
}
#endif
