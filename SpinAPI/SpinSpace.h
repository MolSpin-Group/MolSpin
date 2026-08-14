/////////////////////////////////////////////////////////////////////////
// SpinSpace class (SpinAPI Module)
// ------------------
// Helper class that defines operators/matrices on the space spanned by
// a collection of spins. E.g. the Sz operator in the total Hilbert space.
//
// The SpinSpace class contains methods to produce operators in various
// formats (dense, sparse, Liouville-space-dense, Liouville-space-sparse),
// but for distributed matrix formats use the DistributedSpinSpace class	// TODO: Create DistributedSpinSpace class
// instead (can be constructed from a SpinSpace).
//
// This is meant as a helper class that can be used in the calculation
// tasks, but it is not necessary to use it - e.g. semi-classical methods
// will not be using it. The SpinSpace class is not a friend class of
// any other classes, so every task can be implemented without the use of
// SpinSpace - the SpinSpace class is just here to make life easier.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2019 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////

#ifndef MOD_SpinAPI_SpinSpace
#define MOD_SpinAPI_SpinSpace

#include <vector>
#include <memory>
#include <functional>
#include <random>
#include <string>
#include <armadillo>
#include "SpinAPIDefines.h"
#include "SpinAPIfwd.h"

namespace SpinAPI
{
	struct HilbertRelaxationTerm
	{
		arma::sp_cx_mat L;
		arma::sp_cx_mat R;
		arma::sp_cx_mat M;
		arma::sp_cx_mat L_dag;
		arma::sp_cx_mat R_t;
		arma::sp_cx_mat M_t;
		arma::sp_cx_mat M_dag;
		double rate = 0.0;
	};

	struct HilbertRelaxationDephasingTerm
	{
		arma::sp_cx_mat Psinglet;
		arma::sp_cx_mat Ptriplet;
		arma::sp_cx_mat Psinglet_t;
		arma::sp_cx_mat Ptriplet_t;
		arma::sp_cx_mat Psinglet_dag;
		arma::sp_cx_mat Ptriplet_dag;
		double rate = 0.0;
	};

	struct HilbertRelaxationPhenomenologicalTerm
	{
		// Used by Hilbert-space propagation caches. The corresponding
		// superoperator is rebuilt for the active Hilbert-space dimension.
		double populationRate = 0.0;
		double coherenceRate = 0.0;
	};

	struct HilbertPhenomenologicalRelaxationMap
	{
		// Exact finite-step map in the selected Hamiltonian eigenbasis.
		// Powder tasks prepare this once per orientation and reuse it.
		arma::cx_mat basisToLab;
		arma::cx_mat labToBasis;
		double populationDecay = 1.0;
		double coherenceDecay = 1.0;
	};

	struct HilbertStateRotationCache
	{
		// Spatial powder rotations act on a density matrix through the total
		// angular-momentum generators. Build these operators once per spin space
		// instead of embedding every single-spin operator for every orientation.
		arma::cx_mat Jx;
		arma::cx_mat Jy;
		arma::cx_mat Jz;
		bool rotationInvariant = false;
	};

	enum class HamiltonianApproximation
	{
		Full,
		Secular
	};

	struct HilbertPowderHamiltonian
	{
		// H0 and H1 are kept separate because relaxation bases and rotating-frame
		// propagators use H0 explicitly. Both parts always use the same powder
		// orientation; only H0 is secularized when requested.
		arma::sp_cx_mat H0;
		arma::sp_cx_mat H1;
		arma::sp_cx_mat total;
	};

	enum class TraceSamplingMethod
	{
		SUZ,
		SpinCoherent
	};

	struct HilbertTraceSampleSet
	{
		// Each column is a normalized Hilbert-space state factor. The sample
		// average B B^dagger approaches the normalized density represented by
		// the supplied State, and only omitted spins are trace sampled.
		arma::cx_mat factors;
		arma::uword sampledSubspaceDimension = 0;
	};

	struct HilbertRelaxationCache
	{
		std::vector<HilbertRelaxationTerm> lindblad_terms;
		std::vector<HilbertRelaxationDephasingTerm> dephasing_terms;
		std::vector<HilbertRelaxationPhenomenologicalTerm> phenomenological_terms;
	};

	class SpinSpace
	{
	private:
		// Implementation
		bool useSuperspace;
		bool useFullTensorRotation;
		std::vector<spin_ptr> spins;
		std::vector<interaction_ptr> interactions;
		std::vector<transition_ptr> transitions;
		std::vector<pulse_ptr> pulses;
		double time;				 // The current time to use
		unsigned int trajectoryStep; // The current trajectory step to use
		bool useTrajectoryStep;		 // Set to true if trajectories should be used instead of time, where available
		ReactionOperatorType reactionOperators;

		bool CreateRotationMatrix(double &_alpha, double &_beta, double &_gamma, arma::mat &_R) const;

	public:
		// Constructors / Destructors
		SpinSpace(); // Normal constructors
		explicit SpinSpace(const spin_ptr &);
		explicit SpinSpace(const std::vector<spin_ptr> &);
		explicit SpinSpace(const std::shared_ptr<SpinSystem> &);
		explicit SpinSpace(const SpinSystem &);
		SpinSpace(const SpinSpace &); // Copy-constructor
		~SpinSpace();				  // Destructor

		// Operators
		const SpinSpace &operator=(const SpinSpace &); // Copy-assignment

		// ------------------------------------------------
		// Spin management (SpinSpace_management.cpp)
		// ------------------------------------------------
		// Adding spins to the space
		bool Add(const spin_ptr &);					   // Add the spin to the space
		bool Add(const std::vector<spin_ptr> &);	   // Add all spins in the list to space (spins will not be duplicated)
		bool Add(const std::shared_ptr<SpinSystem> &); // Add all spins and interactions from the SpinSystem (spins will not be duplicated)
		bool Add(const SpinSystem &);
		bool CompleteSpace(const state_ptr &);		 // Add missing spins (if any) to make the space comeplete with respect to the state
		bool CompleteSpace(const interaction_ptr &); // Add missing spins (if any) to make the space comeplete with respect to the interaction

		// Removing spins from the space
		bool Remove(const spin_ptr &);					  // Removes the spin from the space
		bool Remove(const std::vector<spin_ptr> &);		  // Removes all spins in the list from the space
		bool Remove(const std::shared_ptr<SpinSystem> &); // Removes all spins in the SpinSystem from the space
		bool Remove(const SpinSystem &);
		bool RemoveSubspace(const spin_ptr &, const state_ptr &);		// Removes the spin and all spins coupled/entangled with it in the given state from the space
		bool RemoveSubspace(const spin_ptr &, const interaction_ptr &); // Removes the spin and all spins interacting with it in the given interaction from the space
		void ClearSpins();												// Removes all spins from the spin space

		// Checking whether the space contains a range of spins
		bool Contains(const spin_ptr &) const;					  // Checks single spin
		bool Contains(const std::vector<spin_ptr> &) const;		  // Checks whether all spins in range are contained in the space
		bool Contains(const std::shared_ptr<SpinSystem> &) const; // Checks whether all spins in the SpinSystem are contained in the space
		bool Contains(const SpinSystem &) const;
		bool Contains(const std::string SpinObject) const;
		bool ContainsSubspace(const spin_ptr &, const state_ptr &) const;		// Checks whether all spins coupled/entangled with the spin are contained in the space
		bool ContainsSubspace(const spin_ptr &, const interaction_ptr &) const; // Checks whether all spins interacting with the spin are contained in the space

		// ------------------------------------------------
		// Interaction management (SpinSpace_management.cpp)
		// ------------------------------------------------
		// Adding interactions to the space
		bool Add(const interaction_ptr &);						   // Add the interaction to the space
		bool Add(const std::vector<interaction_ptr> &);			   // Add all interactions in the list to space (interactions will not be duplicated)
		bool Remove(const interaction_ptr &);					   // Removes the interaction from the space
		bool Remove(const std::vector<interaction_ptr> &);		   // Removes all interactions in the list from the space
		bool Contains(const interaction_ptr &) const;			   // Checks single interaction
		bool Contains(const std::vector<interaction_ptr> &) const; // Checks whether all interactions in range are contained in the space
		void ClearInteractions();								   // Removes all interactions from the spin space

		// ------------------------------------------------
		// Pulse management (SpinSpace_management.cpp)
		// ------------------------------------------------
		// Adding Pulses to the space
		bool Add(const pulse_ptr &);						   // Add the Pulses to the space
		bool Add(const std::vector<pulse_ptr> &);			   // Add all Pulses in the list to space (Pulses will not be duplicated)
		bool Remove(const pulse_ptr &);					   // Removes the Pulses from the space
		bool Remove(const std::vector<pulse_ptr> &);		   // Removes all Pulses in the list from the space
		bool Contains(const pulse_ptr &) const;			   // Checks single Pulses
		bool Contains(const std::vector<pulse_ptr> &) const; // Checks whether all Pulses in range are contained in the space
		void ClearPulses();								   // Removes all Pulses from the spin space
		
		// ---------------------------------------------
		// Transition management (SpinSpace_management.cpp)
		// ------------------------------------------------
		// Adding transitions to the space
		bool Add(const transition_ptr &);						  // Add the transition to the space
		bool Add(const std::vector<transition_ptr> &);			  // Add all transitions in the list to space (transitions will not be duplicated)
		bool Remove(const transition_ptr &);					  // Removes the transition from the space
		bool Remove(const std::vector<transition_ptr> &);		  // Removes all transitions in the list from the space
		bool Contains(const transition_ptr &) const;			  // Checks single transition
		bool Contains(const std::vector<transition_ptr> &) const; // Checks whether all transitions in range are contained in the space
		void ClearTransitions();								  // Removes all transitions from the spin space

		// ------------------------------------------------
		// Spin state representations in the Hilbert space (SpinSpace_states.cpp)
		// ------------------------------------------------
		arma::cx_mat GetSingleSpinState(const spin_ptr &, int) const;									 // Returns the projection matrix
		bool GetState(const CompleteState &, arma::cx_vec &, bool _useFullBasis = true) const;			 // Vector representing the state
		bool GetState(const state_ptr &, arma::cx_vec &) const;											 // Vector representing the state
		bool GetState(const state_ptr &, arma::cx_mat &) const;											 // Projection operator onto the state (dense matrix)
		bool GetState(const state_ptr &, arma::sp_cx_mat &) const;										 // Projection operator onto the state (sparse matrix)
		bool GetStateSubSpace(const state_ptr &_state, arma::cx_vec &_out) const;						 // Vector representing the state in subspace
		bool FactorizeDensityMatrix(const arma::cx_mat &_density, arma::cx_mat &_factors,
								std::string *_error = nullptr, double _tolerance = 1.0e-10) const; // Build B such that B B^dagger is the normalized density
		bool BuildTraceSamples(const state_ptr &_state, arma::uword _sampleCount, TraceSamplingMethod _method,
						   std::mt19937 &_generator, HilbertTraceSampleSet &_samples,
						   std::string *_error = nullptr) const; // Trace sample only spins omitted from a State object
		bool RotateState(const arma::cx_mat &_state, const arma::mat &_rotation, arma::cx_mat &_out) const; // Rotate a density matrix with the same spatial rotation used for powder averaging
		bool CreateStateRotationCache(const arma::cx_mat &_state, HilbertStateRotationCache &_cache, double _tolerance = 1.0e-12) const; // Precompute total-spin generators and detect rotationally invariant density matrices
		bool CreateStateRotationOperator(const arma::mat &_rotation, const HilbertStateRotationCache &_cache, arma::cx_mat &_operator) const; // Spin-space representation of a molecular-to-lab powder rotation
		bool RotateState(const arma::cx_mat &_state, const arma::mat &_rotation, const HilbertStateRotationCache &_cache, arma::cx_mat &_out) const; // Cached molecular-frame density-matrix rotation
		bool RotateStateFactors(const arma::cx_mat &_factors, const arma::mat &_rotation, const HilbertStateRotationCache &_cache, arma::cx_mat &_out) const; // Rotate pure-state/trace-sampling factors without forming density matrices
		bool PrepareInitialDensityForPowder(const arma::cx_mat &_referenceDensity, const arma::mat &_orientationRotation, StateFrame _stateFrame, bool _discardHamiltonianCoherences, const std::vector<std::string> &_dephasingHamiltonian, const HilbertStateRotationCache *_rotationCache, arma::cx_mat &_orientedDensity); // Apply molecular rotation and optional orientation-specific eigenbasis dephasing
		bool PrepareInitialDensityForPowder(const arma::cx_mat &_referenceDensity, const arma::mat &_orientationRotation, StateFrame _stateFrame, bool _discardHamiltonianCoherences, const std::vector<std::string> &_dephasingHamiltonian, HamiltonianApproximation _dephasingApproximation, const HilbertStateRotationCache *_rotationCache, arma::cx_mat &_orientedDensity); // Explicit full/secular dephasing Hamiltonian selection
		bool DephaseStateInEigenbasis(const arma::cx_mat &_state, const arma::cx_mat &_hamiltonian, arma::cx_mat &_out) const; // Keep only populations in a Hamiltonian eigenbasis
		bool ThermalStateFromHamiltonian(const arma::cx_mat &_hamiltonian, double _Temperature, arma::cx_mat &_mat) const; // Thermal state generated from a specific Hamiltonian
		bool GetThermalState(SpinAPI::SpinSpace &_space, double _Temperature, std::vector<std::string> thermalhamiltonian_list, arma::cx_mat &_mat) const; // Projection operator onto the thermal equilibrium state (dense matrix)

		// ------------------------------------------------
		// Operators in the spin space (SpinSpace_operators.cpp)
		// ------------------------------------------------
		bool CreateOperator(const arma::cx_mat &, const spin_ptr &, arma::cx_mat &) const;				   // Creates an operator in the Hilbert space. The matrix must be square with the multiplicity of the spin as its dimension
		bool OperatorToSuperspace(const arma::cx_mat &, arma::cx_vec &) const;							   // Converts the operator to a vector in the superspace
		bool OperatorFromSuperspace(const arma::cx_vec &, arma::cx_mat &) const;						   // Converts a superspace vector back to a Hilbert space operator
		bool SuperoperatorFromOperators(const arma::cx_mat &, const arma::cx_mat &, arma::cx_mat &) const; // Creates a superspace operator from the two given (left-side and right-side) operators
		bool SuperoperatorFromLeftOperator(const arma::cx_mat &, arma::cx_mat &) const;					   // Assumes right-side operator is identity (more efficient than passing the identity to SuperoperatorFromOperators)
		bool SuperoperatorFromRightOperator(const arma::cx_mat &, arma::cx_mat &) const;				   // Assumes left-side operator is identity (more efficient than passing the identity to SuperoperatorFromOperators)

		// Sparse versions
		bool CreateOperator(const arma::sp_cx_mat &, const spin_ptr &, arma::sp_cx_mat &) const;
		bool OperatorToSuperspace(const arma::sp_cx_mat &, arma::cx_vec &) const;
		bool OperatorFromSuperspace(const arma::cx_vec &, arma::sp_cx_mat &) const;
		bool SuperoperatorFromOperators(const arma::sp_cx_mat &, const arma::sp_cx_mat &, arma::sp_cx_mat &) const;
		bool SuperoperatorFromLeftOperator(const arma::sp_cx_mat &, arma::sp_cx_mat &) const;
		bool SuperoperatorFromRightOperator(const arma::sp_cx_mat &, arma::sp_cx_mat &) const;

		// Re-ordering of the spins, used by the GetState methods when working with entangled states
		// TODO: Consider making non-member non-friend functions, or making such equivalents
		bool ReorderBasis(arma::cx_vec &, const std::vector<spin_ptr> &) const;
		bool ReorderBasis(arma::cx_mat &, const std::vector<spin_ptr> &) const;
		bool ReorderBasis(arma::sp_cx_mat &, const std::vector<spin_ptr> &) const;
		bool ReorderBasis(arma::cx_vec &, const std::vector<spin_ptr> &, const std::vector<spin_ptr> &) const;
		bool ReorderBasis(arma::cx_mat &, const std::vector<spin_ptr> &, const std::vector<spin_ptr> &) const;
		bool ReorderBasis(arma::sp_cx_mat &, const std::vector<spin_ptr> &, const std::vector<spin_ptr> &) const;
		bool ReorderingOperator(arma::sp_cx_mat &, const std::vector<spin_ptr> &, const std::vector<spin_ptr> &) const; // Returns the reordering operator itself, used by the previous methods

		// ------------------------------------------------
		// Spherical tensors (SpinSpace_operators.cpp)
		// ------------------------------------------------

		// Rank 1
		bool Rk1SphericalTensorT0(const spin_ptr &_spin1, const spin_ptr &_spin2, arma::cx_mat &_out) const;
		bool Rk1SphericalTensorTp1(const spin_ptr &_spin1, const spin_ptr &_spin2, arma::cx_mat &_out) const;
		bool Rk1SphericalTensorTm1(const spin_ptr &_spin1, const spin_ptr &_spin2, arma::cx_mat &_out) const;

		// Linear interactions rank 0 & 2
		bool LRk0TensorT0(const spin_ptr &_spin1, const arma::cx_vec &_field, arma::cx_mat &_out) const;
		bool LRk0TensorT0(const spin_ptr &_spin1, const arma::cx_vec &_field, arma::sp_cx_mat &_out) const;
		bool LRk2SphericalTensorT0(const spin_ptr &, const arma::cx_vec &_field, arma::cx_mat &) const;		// T(m=0)
		bool LRk2SphericalTensorT0(const spin_ptr &, const arma::cx_vec &_field, arma::sp_cx_mat &) const;	// T(m=0), sparse
		bool LRk2SphericalTensorTp1(const spin_ptr &, const arma::cx_vec &_field, arma::cx_mat &) const;	// T(m=+1)
		bool LRk2SphericalTensorTp1(const spin_ptr &, const arma::cx_vec &_field, arma::sp_cx_mat &) const; // T(m=+1), sparse
		bool LRk2SphericalTensorTm1(const spin_ptr &, const arma::cx_vec &_field, arma::cx_mat &) const;	// T(m=-1)
		bool LRk2SphericalTensorTm1(const spin_ptr &, const arma::cx_vec &_field, arma::sp_cx_mat &) const; // T(m=-1), sparse
		bool LRk2SphericalTensorTp2(const spin_ptr &, const arma::cx_vec &_field, arma::cx_mat &) const;	// T(m=+2)
		bool LRk2SphericalTensorTp2(const spin_ptr &, const arma::cx_vec &_field, arma::sp_cx_mat &) const; // T(m=+2), sparse
		bool LRk2SphericalTensorTm2(const spin_ptr &, const arma::cx_vec &_field, arma::cx_mat &) const;	// T(m=-2)
		bool LRk2SphericalTensorTm2(const spin_ptr &, const arma::cx_vec &_field, arma::sp_cx_mat &) const; // T(m=-2), sparse

		// Bilinear interactions rank 0 & 2
		bool BlRk0TensorT0(const spin_ptr &_spin1, const spin_ptr &_spin2, arma::cx_mat &_out) const;
		bool BlRk0TensorT0(const spin_ptr &_spin1, const spin_ptr &_spin2, arma::sp_cx_mat &_out) const;
		bool BlRk2SphericalTensorT0(const spin_ptr &, const spin_ptr &, arma::cx_mat &) const;	   // T(m=0)
		bool BlRk2SphericalTensorT0(const spin_ptr &, const spin_ptr &, arma::sp_cx_mat &) const;  // T(m=0), sparse
		bool BlRk2SphericalTensorTp1(const spin_ptr &, const spin_ptr &, arma::cx_mat &) const;	   // T(m=+1)
		bool BlRk2SphericalTensorTp1(const spin_ptr &, const spin_ptr &, arma::sp_cx_mat &) const; // T(m=+1), sparse
		bool BlRk2SphericalTensorTm1(const spin_ptr &, const spin_ptr &, arma::cx_mat &) const;	   // T(m=-1)
		bool BlRk2SphericalTensorTm1(const spin_ptr &, const spin_ptr &, arma::sp_cx_mat &) const; // T(m=-1), sparse
		bool BlRk2SphericalTensorTp2(const spin_ptr &, const spin_ptr &, arma::cx_mat &) const;	   // T(m=+2)
		bool BlRk2SphericalTensorTp2(const spin_ptr &, const spin_ptr &, arma::sp_cx_mat &) const; // T(m=+2), sparse
		bool BlRk2SphericalTensorTm2(const spin_ptr &, const spin_ptr &, arma::cx_mat &) const;	   // T(m=-2)
		bool BlRk2SphericalTensorTm2(const spin_ptr &, const spin_ptr &, arma::sp_cx_mat &) const; // T(m=-2), sparse

		// ------------------------------------------------
		// New added functions for wavefucntion formalism and SSE (by Gediminas Pazera and Luca Gerhards)
		// ------------------------------------------------

		struct return_struct
		{
			arma::cx_colvec result;
			arma::cx_mat PropMat;
			double error_estimate = 0.0;

			operator arma::cx_colvec()
			{
				return result;
			}
		};

		struct return_structMat
		{
			arma::cx_mat result;
			arma::cx_mat phi1;
			arma::cx_mat phi2;
			arma::cx_mat krybasis;
			double error_estimate = 0.0;

			operator arma::cx_mat()
			{
				return result;
			}
		};

		typedef std::function<arma::cx_mat(const arma::sp_cx_mat&, const arma::cx_mat)> GeneratorFunctionMat;
		typedef std::function<arma::cx_vec(const arma::sp_cx_mat&, const arma::cx_vec)> GeneratorFunctionVec;
		
		arma::cx_mat reconstruct_block(const arma::cx_mat& C, const arma::cx_mat& KryBasis, int m, int p);
		arma::cx_mat project_block(const arma::cx_mat& X, const arma::cx_mat& KryBasis, int m, int p);

		arma::cx_colvec SUZstate(const int &spinmult, std::mt19937 &generator);																					 // returns stochastically determined SU(Z) state
		arma::cx_colvec CoherentState(std::vector<SpinAPI::system_ptr>::const_iterator i, std::mt19937 &generator);												 // returns stochastically determined coherent state
		arma::cx_mat HighamProp(arma::sp_cx_mat &H, arma::cx_mat &B, const std::complex<double> t, const std::string precision, arma::mat &M);					 // Propagation method using: https://doi.org/10.1137/100788860
		arma::mat SelectTaylorDegree(const arma::sp_cx_mat &H, const std::string precision, const int lengthB);													 // Precision of Taylor series used for HighamProp
		double normAmEst(const arma::sp_cx_mat &H, double m, std::mt19937 &generator);																			 // Used in SelectTaylorDegree to normalize
		return_struct KrylovExpmGeneral(const arma::sp_cx_mat &H, const arma::cx_colvec &b, const arma::cx_double dt, int KryDim, int HilbSize, bool NEval = false, GeneratorFunctionVec generator = nullptr);				 // Krylov subspace method
		return_struct KrylovExpmSymm(const arma::sp_cx_mat &H, const arma::cx_colvec &b, const arma::cx_double dt, int KryDim, int HilbSize, bool NEval = false);					 // Krylov subspace method for symmetric decay
		bool ArnoldiProcess(const arma::sp_cx_mat &H, const arma::cx_colvec &b, arma::cx_mat &KryBasis, arma::cx_mat &Hessen, int KryDim, double &h_mplusone_m, arma::cx_double dt = arma::cx_double(0.0, 0.0), GeneratorFunctionVec generator = nullptr); // Arnoldi process for propagation using Krylov subsspace
		bool LanczosProcess(const arma::sp_cx_mat &H, const arma::cx_colvec &b, arma::cx_mat &KryBasis, arma::cx_mat &Hessen, int KryDim, double &h_mplusone_m, arma::cx_double dt = arma::cx_double(0.0, 0.0)); // Lanczos process for propagation using Krylov subsspace
		
		return_structMat KrylovExpmGeneral(const arma::sp_cx_mat &H, const arma::cx_mat &b, const arma::cx_double dt, int KryDim, int HilbSize, GeneratorFunctionMat gen, bool NEval = false);
		bool ArnoldiProcess(const arma::sp_cx_mat &H, const arma::cx_mat &b, arma::cx_mat &KryBasis, arma::cx_mat &Hessen, int KryDim, arma::cx_mat &h_mplusone_m, int p, double beta, GeneratorFunctionMat generator, arma::cx_double dt = arma::cx_double(0.0, 0.0));

		//-----------------------------------------------
		// Time Adaptive Versions of the KyrlovPropogation Methods
		//-----------------------------------------------

		struct PropParam
    	{
        	double atol = 1e-8;
        	double rtol = 1e-10;
        	double min = 1e-6;
        	double max = 1e6;
        	double safety = 0.8;
        	double f1 = 0.1;
        	double f2 = 5.0;

        	int max_krylov_iterations = 30;
        	int reject_limit = 2;

			bool dont_evaluate = false;
			double global_error = 0.0;

        	double CurrentTime = 0.0;
			std::vector<double> SetTimePoints;
			unsigned int CurrentTrajectoryStep = 0;
			bool UsePrefactor = false;
			bool UseSetTimePoints = false;
			arma::cx_double TimePrefactor = arma::cx_double(0.0, -1.0);

			bool normalise = true;

			double GetNextTimePoint() {
				if (CurrentTrajectoryStep < SetTimePoints.size())
				{
					double nextTimePoint = SetTimePoints[CurrentTrajectoryStep];
					CurrentTrajectoryStep++;
					return nextTimePoint;
				}
				else
				{
					CurrentTrajectoryStep = 0; // Reset trajectory step if we exceed the number of time points
					return std::numeric_limits<double>::infinity(); // Return infinity to indicate no more time
				}
			}

			void ResetTrajectory() {
				CurrentTrajectoryStep = 0;
			}
    	};
    	
		struct TimePropReturnInfo
    	{
        	double timestep;
			double timestep_used;
			bool step_accepted;
			arma::cx_colvec result;
		};

		
		/// @brief Return struct from time-adaptive propogators
		/// @param timestep - the proposed new timestep for the following step
		/// @param timestep_used - the timestep actually used in that step (this may differ from the one provided)
		/// @param step_accepted - if the step was accepted on the first try - not always populted
		/// @param result - the result of the propogation (sometimes the density matrix but not always)
		/// @param krybasis - not always the krybasis but occasially just a random matrix that needs to be returned
		struct TimePropReturnInfoMat
    	{
        	double timestep;
			double timestep_used;
			bool step_accepted;
			arma::cx_mat result;

			arma::cx_mat phi1;
			arma::cx_mat phi2;
			arma::cx_mat phi3;
			arma::cx_mat krybasis;
		};

		struct TimeAdaptiveKrylovCache
		{
			int KrylovDim = 0;
			// Fixed-dimension Krylov calls do not truncate their requested
			// basis. Adaptive propagators set and clear this tolerance.
			double KrylovDimTol = 0.0;
		};

		double Adjusth(double R, double safety, double f1, double f2, double h, int order = 4);

		TimePropReturnInfo TimeAdapativeKrylovRoutine(const arma::sp_cx_mat &H, const arma::cx_colvec &b, arma::cx_double dt, int kryDim, int HilbSize, PropParam &propParam, bool general, bool reset = true);
		TimePropReturnInfo TimeAdaptiveKrylovGeneral(const arma::sp_cx_mat &H, const arma::cx_colvec &b, arma::cx_double dt, int kryDim, int HilbSize, PropParam &propParam, bool reset = true);
		TimePropReturnInfo TimeAdaptiveKrylovSymm(const arma::sp_cx_mat &H, const arma::cx_colvec &b, arma::cx_double dt, int kryDim, int HilbSize, PropParam &propParam, bool reset = true);

		TimePropReturnInfoMat TimeAdapativeKrylovRoutine(const arma::sp_cx_mat &H, const arma::cx_mat &b, arma::cx_double dt, int kryDim, int HilbSize, PropParam &propParam, bool general, GeneratorFunctionVec gen, bool reset = true);
		TimePropReturnInfoMat TimeAdaptiveKrylovGeneral(const arma::cx_mat &H, const arma::cx_mat &b, arma::cx_double dt, int kryDim, int HilbSize, PropParam &propParam, bool reset = true, GeneratorFunctionVec gen = nullptr);

		typedef std::function<void(std::vector<arma::sp_cx_mat>&, std::vector<arma::cx_mat>)> NonLinearTermEval;
		struct CachedInfo {
			arma::cx_mat expH;
			arma::cx_mat phi1;
			arma::cx_mat phi2;
			arma::cx_mat phi3;

			arma::cx_double prev_timestep;
		};

		std::vector<TimePropReturnInfoMat> ETD2RK_exponential(const std::vector<arma::cx_mat> &H, const std::vector<arma::cx_mat> &b, arma::cx_double dt, PropParam &prop, NonLinearTermEval NLfunc, std::vector<bool> reval = {true}, std::vector<CachedInfo> cache = {});

		// ------------------------------------------------
		// Hamiltonian representations in the space (SpinSpace_hamiltonians.cpp)
		// ------------------------------------------------
		bool InteractionOperator(const interaction_ptr &, arma::cx_mat &) const;	// Returns the matrix representation of the interaction on the spin space (dense matrix)
		bool InteractionOperator(const interaction_ptr &, arma::sp_cx_mat &) const; // Returns the matrix representation of the interaction on the spin space (sparse matrix)
		bool InteractionOperatorRotatedZYZ(const interaction_ptr &, arma::mat &, arma::sp_cx_mat &) const;
		bool InteractionOperatorRotated_SA(const interaction_ptr &, arma::mat &, arma::sp_cx_mat &) const;
		bool Hamiltonian(arma::cx_mat &, int TaskNum = 0) const;										// Total Hamiltonian operator (dense matrix)
		bool Hamiltonian(arma::sp_cx_mat &, int TaskNum = 0) const;									// Total Hamiltonian operator (sparse matrix)
		bool SemiClassicalHamiltonian(arma::sp_cx_mat &, std::vector<interaction_ptr>&) const; 					// SemiClassical approximation of the Hamiltonian (sparse matrix), refer to BasicTask.cpp for tasknum conversion
		bool SemiClassicalHamiltonian(arma::cx_mat &, std::vector<interaction_ptr>&) const; 					// SemiClassical approximation of the Hamiltonian (dense matrix), refer to BasicTask.cpp for tasknum conversion 	 					
		bool StaticHamiltonian(arma::cx_mat &) const;								// Time-independent part of the Hamiltonian operator (dense matrix)
		bool StaticHamiltonian(arma::sp_cx_mat &) const;							// Time-independent part of the Hamiltonian operator (sparse matrix)
		bool DynamicHamiltonian(arma::cx_mat &) const;								// Time-dependent part of the Hamiltonian operator (dense matrix)
		bool DynamicHamiltonian(arma::sp_cx_mat &) const;							// Time-dependent part of the Hamiltonian operator (sparse matrix)
		bool ThermalHamiltonian(std::vector<std::string> thermalhamiltonian_list, arma::cx_mat &_out) const;							// Time-independent part of the Hamiltonian for thermal state (dense matrix)
		bool ThermalHamiltonian(std::vector<std::string> thermalhamiltonian_list, arma::sp_cx_mat &_out) const;							// Time-independent part of the Hamiltonian for thermal state (sparse matrix)
		bool StaticHamiltonianRotatedZYZ(const arma::mat &rotmatrix, arma::sp_cx_mat &_out) const; // All static interactions after a common molecular-to-lab rotation
		bool StaticHamiltonianRotatedSA(const arma::mat &rotmatrix, arma::sp_cx_mat &_out) const;  // All static interactions in the high-field secular approximation
		bool DynamicHamiltonianRotatedZYZ(const arma::mat &rotmatrix, arma::sp_cx_mat &_out) const; // Active time-dependent interactions after the same powder rotation
		bool BaseHamiltonianRotatedZYZ(std::vector<std::string> basehamiltonian_list, arma::mat rotmatrix, arma::sp_cx_mat &_out) const;
		bool BaseHamiltonianRotated_SA(std::vector<std::string> basehamiltonian_list, arma::mat rotmatrix, arma::sp_cx_mat &_out) const;
		bool PowderHamiltonianRotated(const std::vector<std::string> &h0list, const std::vector<std::string> &h1list, const arma::mat &rotmatrix, HamiltonianApproximation approximation, HilbertPowderHamiltonian &hamiltonian) const;
		bool PowderHamiltonianRotatedSA(const std::vector<std::string> &h0list, const std::vector<std::string> &h1list, const arma::mat &rotmatrix, arma::sp_cx_mat &H0, arma::sp_cx_mat &H1, arma::sp_cx_mat &H) const;

		// ------------------------------------------------
		// Transitions/decay operators (SpinSpace_transitions.cpp)
		// ------------------------------------------------
		// NOTE: Only works with Haberkorn operators! TODO: Consider implementations for other reaction operator types.
		// Provides the operator "k/2 * P" in Hilbert space, where P is the projection onto a state
		// Provides the anti-commutator operator "k/2 * {P,.}" in superoperator space
		bool ReactionOperator(const transition_ptr &, arma::cx_mat &, const ReactionOperatorType &_forcedReactionOperatorType = ReactionOperatorType::Unspecified) const;	 // The last parameter can be used to force the use of a specific reaction operator type,
		bool ReactionOperator(const transition_ptr &, arma::sp_cx_mat &, const ReactionOperatorType &_forcedReactionOperatorType = ReactionOperatorType::Unspecified) const; // but by default the reaction operator type of the transition or the spinspace will be used.
		bool TotalReactionOperator(arma::cx_mat &, const ReactionOperatorType &_forcedReactionOperatorType = ReactionOperatorType::Unspecified) const;						 // Total reaction operator (dense matrix)
		bool TotalReactionOperator(arma::sp_cx_mat &, const ReactionOperatorType &_forcedReactionOperatorType = ReactionOperatorType::Unspecified) const;					 // Total reaction operator (sparse matrix)
		bool StaticTotalReactionOperator(arma::cx_mat &, const ReactionOperatorType &_forcedReactionOperatorType = ReactionOperatorType::Unspecified, bool NoInterSystem = false) const;				 // Time-independent part of the total reaction operator (dense matrix)
		bool StaticTotalReactionOperator(arma::sp_cx_mat &, const ReactionOperatorType &_forcedReactionOperatorType = ReactionOperatorType::Unspecified) const;				 // Time-independent part of the total reaction operator (sparse matrix)
		bool DynamicTotalReactionOperator(arma::cx_mat &, const ReactionOperatorType &_forcedReactionOperatorType = ReactionOperatorType::Unspecified) const;				 // Time-dependent part of the total reaction operator (dense matrix)
		bool DynamicTotalReactionOperator(arma::sp_cx_mat &, const ReactionOperatorType &_forcedReactionOperatorType = ReactionOperatorType::Unspecified) const;			 // Time-dependent part of the total reaction operator (sparse matrix)
		ReactionOperatorType GetReactionOperatorType() const;																												 // Returns the reaction operator type used by the SpinSpace (superspace only)

		// Methods to create reaction operators in the target spin system (i.e. for creation), where the 'double' describes the amount of source state in the source system
		bool ReactionTargetOperator(const transition_ptr &, double, arma::cx_mat &) const;
		bool ReactionTargetOperator(const transition_ptr &, double, arma::sp_cx_mat &) const;
		bool ReactionSourceOperator(const transition_ptr &, double, arma::cx_mat &) const;
		bool ReactionSourceOperator(const transition_ptr &, double, arma::sp_cx_mat &) const;

		// ------------------------------------------------
		// Relaxation operators (SpinSpace_relaxation.cpp)
		// ------------------------------------------------
		// NOTE: Dense/sparse operators require superspace; use HilbertRelaxationCache for Hilbert-space propagation.
		bool RelaxationOperator(const operator_ptr &, arma::cx_mat &) const;
		bool RelaxationOperator(const operator_ptr &, arma::sp_cx_mat &) const;
		bool RelaxationOperator(const operator_ptr &, HilbertRelaxationCache &) const;
		bool ApplyRelaxationHilbert(const HilbertRelaxationCache &, const arma::cx_mat &, arma::cx_mat &) const;
		bool RelaxationSuperoperatorHilbert(const HilbertRelaxationCache &, arma::cx_mat &) const;
		// Hilbert-space powder tasks rebuild anisotropic spin operators for
		// each spatial orientation. Phenomenological relaxation is applied
		// directly to matrix elements in the selected Hamiltonian eigenbasis.
		bool PowderRelaxationOperatorHilbert(const operator_ptr &, const arma::mat &, HilbertRelaxationCache &) const;
		// Differential form used when phenomenological terms must be combined
		// with explicit relaxation channels in an RK propagation fallback.
		bool ApplyPhenomenologicalRelaxationHilbert(const std::vector<HilbertRelaxationPhenomenologicalTerm> &, const arma::cx_mat &, const arma::cx_mat &, arma::cx_mat &) const;
		// Exact finite-step form used by phenomenological-only Hilbert-space
		// propagation. Preparation is separated from application so a powder
		// task can reuse basis transforms and decay factors across time steps.
			bool CreatePhenomenologicalRelaxationMapHilbert(const std::vector<HilbertRelaxationPhenomenologicalTerm> &, const arma::cx_mat &, double, HilbertPhenomenologicalRelaxationMap &) const;
			bool ApplyPhenomenologicalRelaxationMapHilbert(const HilbertPhenomenologicalRelaxationMap &, arma::cx_mat &, arma::cx_mat &) const;
			bool ApplyPhenomenologicalRelaxationMapInBasisHilbert(const HilbertPhenomenologicalRelaxationMap &, arma::cx_mat &) const; // Exact element-wise map when propagation already uses the selected eigenbasis
		// Superspace representation retained for steady-state timeinf solves.
		bool PhenomenologicalRelaxationSuperoperatorHilbert(const std::vector<HilbertRelaxationPhenomenologicalTerm> &, const arma::cx_mat &, arma::cx_mat &) const;

		// Same relaxation operators, but when unitary transformation of projection operators is required
		bool RelaxationOperatorFrameChange(const operator_ptr &_operator, arma::cx_mat _rotationmatrix, arma::cx_mat &_out) const;
		bool RelaxationOperatorFrameChange(const operator_ptr &_operator, arma::cx_mat _rotationmatrix, arma::sp_cx_mat &_out) const;
		bool RelaxationOperatorFrameChange(const operator_ptr &_operator, arma::sp_cx_mat _rotationmatrix, arma::sp_cx_mat &_out) const;
		// Powder tasks supply two rotations: an optional spatial powder-frame
		// rotation for molecule-fixed relaxation axes, followed by the usual
		// Hilbert-basis change. Lab-frame axes intentionally skip the first.
		bool RelaxationOperatorFrameChangeRotated(const operator_ptr &_operator, arma::cx_mat _rotationmatrix, arma::mat _spatialrotation, arma::cx_mat &_out) const;
		bool RelaxationOperatorFrameChangeRotated(const operator_ptr &_operator, arma::cx_mat _rotationmatrix, arma::mat _spatialrotation, arma::sp_cx_mat &_out) const;
		// Powder relaxation helper: construct every supported relaxation operator
		// in the orientation-specific H0 eigenbasis after applying the spatial
		// powder rotation. The second overload transforms that eigenbasis
		// superoperator back to the current propagation basis.
		bool PowderRelaxationOperatorEigenbasis(const operator_ptr &_operator, arma::cx_mat _eigenvectors, arma::mat _spatialrotation, arma::cx_mat &_out) const;
		bool PowderRelaxationOperatorEigenbasis(const operator_ptr &_operator, arma::cx_mat _eigenvectors, arma::mat _spatialrotation, arma::sp_cx_mat &_out) const;
		bool PowderRelaxationOperator(const operator_ptr &_operator, arma::cx_mat _eigenvectors, arma::mat _spatialrotation, arma::cx_mat &_out) const;
		bool PowderRelaxationOperator(const operator_ptr &_operator, arma::cx_mat _eigenvectors, arma::mat _spatialrotation, arma::sp_cx_mat &_out) const;

		// ------------------------------------------------
		// Pulse operators (SpinSpace_pulses.cpp)
		// ------------------------------------------------
		bool PulseOperator(const pulse_ptr &_pulse, arma::cx_mat &_out) const;
		bool PulseOperator(const pulse_ptr &_pulse, arma::sp_cx_mat &_out) const;
		bool PulseOperator_mw(const pulse_ptr &_pulse, arma::cx_mat &_out, double &_time) const;
		bool PulseOperator_mw(const pulse_ptr &_pulse, arma::sp_cx_mat &_out, double &_time) const;
		bool PulseOperator(const pulse_ptr &_pulse, arma::cx_mat &_left, arma::cx_mat &_right) const;
		bool PulseOperator(const pulse_ptr &_pulse, arma::sp_cx_mat &_left, arma::sp_cx_mat &_right) const;
		bool PulseOperatorFrameChange(const pulse_ptr &_pulse, arma::cx_mat _rotationmatrix, arma::cx_mat &_out) const;
		bool PulseOperatorFrameChange(const pulse_ptr &_pulse, arma::sp_cx_mat _rotationmatrix, arma::sp_cx_mat &_out) const;
		bool PulseOperatorFrameChange_mw(const pulse_ptr &_pulse, arma::cx_mat _rotationmatrix, arma::cx_mat &_out, double &_time) const;
		bool PulseOperatorFrameChange_mw(const pulse_ptr &_pulse, arma::sp_cx_mat _rotationmatrix, arma::sp_cx_mat &_out, double &_time) const;
		bool PulseOperatorFrameChange(const pulse_ptr &_pulse, arma::cx_mat _rotationmatrix, arma::cx_mat &_left, arma::cx_mat &_right) const;
		bool PulseOperatorOnStatevector(const pulse_ptr &_pulse, arma::cx_mat &_out) const;
		bool PulseOperatorOnStatevector(const pulse_ptr &_pulse, arma::sp_cx_mat &_out) const;
		bool CreateRotAngle(double _angle_deg, double &result) const;

		// ------------------------------------------------
		// Other public methods
		// ------------------------------------------------
		unsigned int SpaceDimensions() const;		 // Returns the size of the spin space, depending on whether superspace is used or not
		unsigned int HilbertSpaceDimensions() const; // Returns the size of the spin space Hilbert space
		unsigned int SuperSpaceDimensions() const;	 // Returns the size of the spin space super-space
		bool HasTimedependentInteractions() const;
		bool HasTimedependentTransitions() const;

		// ------------------------------------------------
		// Settings for the spin space
		// ------------------------------------------------
		bool UseSuperoperatorSpace(bool);							// Determines whether all returned operators, etc. will be in superoperator-/Liouville-space
		bool UseFullTensorRotation(bool);							// Controls whether powder-rotated tensors keep off-diagonal elements
		bool SetReactionOperatorType(const ReactionOperatorType &); // Sets the type of reaction operator to be produced - NOTE: Only works in superspace
		bool SetTime(double);										// Set the current time, used to set states from trajectories (provided the trajectories have "time" columns, otherwise first step is used)
		bool SetTrajectoryStep(unsigned int);						// Set the current step to be used in all trajectories (trajectories with too few steps will use last step)
	private:
		bool InternalCreateSCCompositeMatrix(const SpinAPI::interaction_ptr&, int, arma::sp_cx_mat&) const;
		bool InternalCreateSCCompositeMatrix(const SpinAPI::interaction_ptr&, int, arma::cx_mat&) const;
		bool SCSupportedTasks(int tasknum) const; //refer to BasicTask.cpp for tasknum conversion 
		bool CreateSpinOperatorTriplet(const spin_ptr &_spin, arma::cx_mat &_Sx, arma::cx_mat &_Sy, arma::cx_mat &_Sz) const;
		bool CreateSpinOperatorTriplet(const spin_ptr &_spin, arma::sp_cx_mat &_Sx, arma::sp_cx_mat &_Sy, arma::sp_cx_mat &_Sz) const;
		template <typename MatrixType>
		bool RotateCartesianOperatorTriplet(const arma::mat &_spatialrotation, MatrixType &_Sx, MatrixType &_Sy, MatrixType &_Sz) const;
		template <typename MatrixType>
		void TransformOperatorToBasis(const arma::cx_mat &_basisrotation, MatrixType &_operator) const;
		template <typename MatrixType>
		bool CreateRotatedSpinTripletInBasis(const spin_ptr &_spin, const arma::cx_mat &_basisrotation, const arma::mat &_spatialrotation, MatrixType &_Sx, MatrixType &_Sy, MatrixType &_Sz) const;
		template <typename MatrixType>
		bool CreateRotatedSpinPlusMinusInBasis(const spin_ptr &_spin, const arma::cx_mat &_basisrotation, const arma::mat &_spatialrotation, MatrixType &_Splus, MatrixType &_Sminus) const;
		template <typename MatrixType>
		bool RelaxationOperatorFrameChangeRotatedInternal(const operator_ptr &_operator, const arma::cx_mat &_basisrotation, const arma::mat &_spatialrotation, MatrixType &_out) const;
		bool RelaxationOperatorHilbertInternal(const operator_ptr &, const arma::mat *, HilbertRelaxationCache &) const;

	};

	// Non-member non-friend functions
	// ------------------------------------------------
	// Creation operators (SpinSpace_transitions.cpp)
	// ------------------------------------------------
	// The creation operators have the form C = |a><b| where |b> is a state in the source system,
	// and |a> is a state in the target system. Thus this operator transforms state |b> of the source
	// system into state |a> of the target system.
	// NOTE: Creation operators do not contain the rate constant.
	bool CreationOperator(const transition_ptr &, const SpinSpace &, const SpinSpace &, arma::cx_mat &, bool _useSuperoperatorSpace = false);
	bool CreationOperator(const transition_ptr &, const SpinSpace &, const SpinSpace &, arma::sp_cx_mat &, bool _useSuperoperatorSpace = false);
}

#endif
