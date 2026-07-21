/////////////////////////////////////////////////////////////////////////
// TaskStaticHSDirectSpectra (RunSection module) by Luca Gerhards
// ------------------
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_TaskStaticHSDirectSpectra
#define MOD_RunSection_TaskStaticHSDirectSpectra

#include <armadillo>
#include <tuple>
#include <string>
#include <vector>
#include "BasicTask.h"
#include "PowderGrid.h"
#include "SpinAPIDefines.h"
#include "SpinSpace.h"

namespace RunSection
{
	class TaskStaticHSDirectSpectra : public BasicTask
	{
	private:
		struct DensityPropagationPlan
		{
			bool useCompactFreeEvolutionMap = false;
			arma::uword hilbertDimension = 0;
			arma::uword densityDimension = 0;
			double denseMapMiB = 0.0;
			std::string reason;
		};

		struct DetectionOperatorSet
		{
			// Operators used for output expectations. Sparse and dense forms
			// share the same column order; vectorized form is used by compact
			// density-map propagation.
			std::vector<arma::sp_cx_mat> sparse;
			std::vector<arma::cx_mat> dense;
			std::vector<arma::cx_vec> vectorized;
			bool useSparse = false;
		};

		double timestep;
		double totaltime;
		bool powderFullSphere;

		SpinAPI::ReactionOperatorType reactionOperators;

		void WriteHeader(std::ostream &); // Write header for the output file
		static double TraceSparseDense(const arma::sp_cx_mat &_A, const arma::cx_mat &_B);
		static double TraceDenseDense(const arma::cx_mat &_A, const arma::cx_mat &_B);
		static void WriteTransitionYieldHeader(const SpinAPI::system_ptr &_system, std::ostream &_stream);
		static bool BuildInitialDensityMatrix(const SpinAPI::system_ptr &_system, SpinAPI::SpinSpace &_space, arma::cx_mat &_rho0, std::ostream &_logstream);
		static arma::cx_mat FactorizeDensityMatrix(const arma::cx_mat &_rho0, std::ostream &_logstream);
		static bool AddPhenomenologicalTerm(const SpinAPI::operator_ptr &_relaxationOperator, std::vector<SpinAPI::HilbertRelaxationPhenomenologicalTerm> &_terms);
		static bool DiagonalizeRelaxationBasis(const arma::sp_cx_mat &_basisHamiltonian, arma::cx_mat &_basisEigenvectors, std::ostream &_logstream);
		static DensityPropagationPlan EvaluateDensityPropagationPlan(arma::uword _hilbertDimension, int _numSteps, bool _methodTimeEvo, bool _splitExpmEnabled, bool _freeEvolutionIsTimeIndependent);
		bool BuildDetectionOperators(const SpinAPI::system_ptr &_system, SpinAPI::SpinSpace &_space, bool _cidsp, arma::uword _hilbertDimension, DetectionOperatorSet &_operators, std::ostream &_logstream) const;
		bool CreateRotationMatrix(double &_alpha, double &_beta, double &_gamma, arma::mat &_R) const;
		bool CreateUniformGrid(int &_Npoints, SpinAPI::PowderGrid &_uniformGrid) const;
		bool CreateExplicitPowderGrid(SpinAPI::PowderGrid &_grid);

	protected:
		bool RunLocal() override;
		bool Validate() override;

	public:
		// Constructors / Destructors
		TaskStaticHSDirectSpectra(const MSDParser::ObjectParser &, const RunSection &); // Normal constructor
		~TaskStaticHSDirectSpectra();												   // Destructor
	};
}

#endif
