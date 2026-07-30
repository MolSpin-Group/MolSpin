/////////////////////////////////////////////////////////////////////////
// TaskMultiStaticSS(RunSection module)
// ------------------
//
// Integrated steady-state calculation in Liouville space for a reaction
// network that can transfer population between several SpinSystems.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_TaskMultiStaticSS
#define MOD_RunSection_TaskMultiStaticSS

#include "BasicTask.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "SpinAPIDefines.h"
#include "PowderGrid.h"

namespace RunSection
{
	class TaskMultiStaticSS : public BasicTask
	{
	private:
		enum class HamiltonianMode
		{
			FullFixed,
			RotatedFull,
			RotatedSecular
		};

		enum class LinearSolver
		{
			Automatic,
			Sparse,
			Dense
		};

		struct SystemContext
		{
			SpinAPI::system_ptr system;
			std::shared_ptr<SpinAPI::SpinSpace> space;
			arma::uword offset = 0;
			arma::uword superDimension = 0;
			std::vector<SpinAPI::state_ptr> initialStates;
			std::vector<double> initialWeights;
			SpinAPI::StateFrame initialFrame = SpinAPI::StateFrame::Fixed;
			bool dephaseInitialState = false;
			double initialPopulation = 1.0;
		};

		bool productYieldsOnly;
		SpinAPI::ReactionOperatorType reactionOperators;
		HamiltonianMode hamiltonianMode;
		LinearSolver linearSolver;
		SpinAPI::PowderGridDomain powderDomain;
		std::string powderGridType;
		std::string powderSymmetry;
		int powderSamplingPoints;
		int powderGridSize;
		bool normalizePowderWeights;
		bool diagnostics;
		double solverResidualTolerance;
		unsigned int denseSolverThreshold;

		void WriteHeader(std::ostream &);
		bool CreatePowderGrid(SpinAPI::PowderGrid &, bool &);
		bool PrepareSystemContexts(std::vector<SystemContext> &, arma::uword &);
		bool BuildStaticHamiltonian(SystemContext &, const arma::mat &, arma::sp_cx_mat &);
		bool PrepareInitialDensity(SystemContext &, const arma::mat &, const arma::sp_cx_mat &, arma::cx_mat &);
		bool BuildReactionLoss(SystemContext &, const arma::mat &, arma::sp_cx_mat &);
		bool BuildCreationOperator(const SpinAPI::transition_ptr &, SystemContext &, SystemContext &, const arma::mat &, arma::sp_cx_mat &);
		bool BuildLiouvillian(std::vector<SystemContext> &, const arma::mat &, arma::sp_cx_mat &, arma::cx_vec &);
		bool SolveIntegratedDensity(const arma::sp_cx_mat &, const arma::cx_vec &, arma::cx_vec &);
		bool ProjectOrientation(std::vector<SystemContext> &, const arma::mat &, const arma::cx_vec &, std::vector<double> &);
		bool OrientedStateProjector(SystemContext &, const SpinAPI::state_ptr &, const arma::mat &, SpinAPI::StateFrame, arma::cx_mat &);
		bool OrientedStateVector(SystemContext &, const SpinAPI::state_ptr &, const arma::mat &, SpinAPI::StateFrame, arma::cx_vec &);
		SpinAPI::StateFrame SystemStateFrame(const SystemContext &, const std::string &, SpinAPI::StateFrame) const;
		SpinAPI::StateFrame TransitionStateFrame(const SpinAPI::transition_ptr &, const SystemContext &, bool) const;
		SpinAPI::StateFrame ObservableStateFrame(const SystemContext &, const SpinAPI::state_ptr &) const;
		SpinAPI::StateFrame ParseStateFrame(const std::string &, SpinAPI::StateFrame) const;
		double RealProjection(const arma::cx_double &, const std::string &, bool &);
		const char *HamiltonianModeName() const;

	protected:
		bool RunLocal() override;
		bool Validate() override;

	public:
		// Constructors / Destructors
		TaskMultiStaticSS(const MSDParser::ObjectParser &, const RunSection &); // Normal constructor
		~TaskMultiStaticSS();												   // Destructor
	};
}

#endif
