/////////////////////////////////////////////////////////////////////////
// Interaction class (SpinAPI Module)
// ------------------
// Basic interaction class.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_SpinAPI_Interaction
#define MOD_SpinAPI_Interaction

#include <vector>
#include <functional> 
#include <array>
#include <memory>
#include <armadillo>
#include "SpinAPIDefines.h"
#include "MSDParserfwd.h"
#include "ActionTarget.h"
#include "Tensor.h"
#include "SpinAPIfwd.h"

namespace SpinAPI
{
	typedef std::tuple<SCMatrix3x3,int,double> SCHyperfineField;
	typedef std::function<double (std::array<double,3>)> SCDistributionF;

	enum class SCDistribution
	{
		FJC = 0, //Freely jointed chain
		DEFUALT //is FJC
	};

	std::ostream &operator<<(std::ostream&, const SCMatrix3x3&);
	
	class Interaction
	{
	private:
		// Data members
		std::shared_ptr<MSDParser::ObjectParser> properties; // Use a pointer to the object to minimize compilation dependencies
		std::shared_ptr<Tensor> couplingTensor;				 // Coupling tensor for two-spin interactions, i.e. "A" in "S1 * A * S2". Example: Hyperfine tensor. Could also be used for one-spin interactions.

		arma::vec field;				// Field vector for one-spin interactions, i.e. "B" in "S1 * B". Example: Magnetic field in Zeeman interaction.
		double dvalue, evalue;			// D and E value for zero-field splitting
		bool EnergyShift;
		std::vector<spin_ptr> group1;	// Spins to use for one-spin interaction and left-hand-side of two-spin interaction
		std::vector<spin_ptr> group2;	// Spins to use on right-hand-side of coupling tensor in two-spin interaction
		InteractionType type;			// Interaction type (one-spin / two-spin)
		InteractionFieldType fieldType; // Field type for one-spin interactions (static / time-dependence specification)
		double prefactor;				// An optional additional prefactor that can be specified in input file (default = 1.0)
		bool addCommonPrefactor;		// Whether or not to multiply by "g mu_B" for electronic spins, or the equivalent for nuclear spins
		bool ignoreTensors;
		bool isValid; // If false, it overrides the existing IsValid function and forces it to automatically return false. Used when larger spin systems are split into smaller spin systems and the interaction hasn't been assigned to a given spin system

		// Trajectory parameters
		Trajectory trajectory;
		bool trjHasTime;
		bool trjHasField;
		bool trjHasTensor;
		bool trjHasPrefactor;
		unsigned int trjTime;
		unsigned int trjFieldX;
		unsigned int trjFieldY;
		unsigned int trjFieldZ;
		unsigned int trjPrefactor;

		// Special data members for time-dependent fields
		double tdFrequency;
		double tdPhase;
		arma::vec tdAxis;
		bool tdPerpendicularOscillation;
		arma::vec tdInitialField; // Time-dependent fields will have readonly ActionTargets, so we can save the initial state

		// Special data members for time-dependent tensors
		InteractionTensorType tensorType; // Tensor type for tensor interactions (static / time-dependence specification)
		double tdTimestep;
		arma::mat tdInitialTensor;
		double tdMinFreq;
		double tdMaxFreq;
		arma::mat tdFreqs;
		arma::mat tdAmps;
		arma::mat tdPhases;
		int tdComponents;
		bool tdRandOrients;
		arma::mat tdThetas;
		arma::mat tdPhis;
		double tdCorrTime;
		double tdAmp;
		bool tdPrintTensor;
		bool tdPrintField;

		//Special data members for Semi-Classical
		std::vector<SCHyperfineField> hffield;
		unsigned int orientations;
		std::vector<double> OriWeights;
		std::vector<double> BondLengths;
		std::vector<double> Spacing;
		double tau;
		SCDistribution dist;

		// Special data members for random number generation
		int tdSeed;
		bool tdAutoseed;
		std::mt19937 tdGenerator;

		// Special data members for tensor with specified rotation frame
		std::vector<double> framelist;

		// Private methods to create ActionTargets
		std::vector<RunSection::NamedActionVector> CreateActionVectors(const std::string &);
		std::vector<RunSection::NamedActionScalar> CreateActionScalars(const std::string &);

		// Member functions for time-dependent tensors
		void TensorTimeDependenceMonochromatic(arma::mat, double, double, double);
		void TensorTimeDependenceBroadband(arma::mat, double, arma::mat, arma::mat, arma::mat, int);
		void TensorTimeDependenceOUGeneral(arma::mat, double, double, double);

		// Helper method called by ParseSpinGroups
		bool AddSpinList(const std::string &, const std::vector<spin_ptr> &, std::vector<spin_ptr> &, const std::vector<spin_ptr> *_crossCheck = nullptr);

	public:
		// Constructors / Destructors
		Interaction(std::string, std::string); // Normal constructor
		Interaction(const Interaction &);	   // Copy-constructor
		~Interaction();						   // Destructor

		// Operators
		const Interaction &operator=(const Interaction &); // Copy-assignment

		// Name and validation
		std::string Name() const;
		bool IsValid() const;
		void SetValid(bool valid)
		{
			isValid = valid;
		};

		// Read spins into group1 and group2, returns false if some spins were not found
		bool ParseSpinGroups(const std::vector<spin_ptr> &);

		// Get interaction information
		std::vector<spin_ptr> Group1() const { return this->group1; };
		std::vector<spin_ptr> Group2() const { return this->group2; };
		InteractionType Type() const { return this->type; };
		InteractionFieldType FieldType() const { return this->fieldType; };
		InteractionTensorType TensorType() const { return this->tensorType; };
		const bool AddCommonPrefactor() const { return this->addCommonPrefactor; };
		const bool IgnoreTensors() const { return this->ignoreTensors; };
		const double Prefactor() const;
		const std::shared_ptr<const Tensor> CouplingTensor() const { return this->couplingTensor; };
		bool SetField(const arma::vec &);
		const arma::vec Field() const;
		const double Dvalue() const;
		const double Evalue() const;
		const bool ES() const {return this->EnergyShift; };
		//const double Hfiamplitude() const;
		const std::vector<SCHyperfineField> Hfiamplitude() const;
		const int Orientations() const;
		std::vector<double>& GetOriWeights() { return this->OriWeights; };
		std::vector<double>& GetSpacing() { return this->Spacing; };
		const std::vector<double> VL() const { return this->BondLengths; }
		bool HasFieldTimeDependence() const;
		bool HasTensorTimeDependence() const;
		bool HasTimeDependence() const;
		const arma::vec Framelist() const;

		// Get time-dependency parameters
		double GetTDFrequency() const { return this->tdFrequency; };
		double GetTDPhase() const { return this->tdPhase; };

		// Trajectory handling
		unsigned int TrajectoryLength() const; // Number of steps in the trajectory (0 means no trajectory)
		bool SetTrajectoryStep(unsigned int);  // Set parameters from trajectory based on step number
		bool SetTime(double);				   // Set parameters from trajectory or time-dependence function based on time

		// Allow access to custom properties to be used for custom tasks
		std::shared_ptr<const MSDParser::ObjectParser> Properties() const;

		// Public methods
		std::vector<spin_ptr> CompleteSet(const spin_ptr &) const; // Returns the minimal list of spins containing the given spin that is complete with respect to the interaction
		bool CompleteSet(std::vector<spin_ptr> &) const;		   // Extends list to the minimal list of spins containing all the spins in the given list, and is complete with respect to the interaction
		bool IsComplete(const std::vector<spin_ptr> &) const;	   // Checks whether any spins in the vector are interacting with any spins outside the vector

		// Public method for creating ActionTargets
		void GetActionTargets(std::vector<RunSection::NamedActionScalar> &, std::vector<RunSection::NamedActionVector> &, const std::string &);

		//get distribution functiom
		SCDistributionF f;
	};

	// Define alias for interaction-pointers
	using interaction_ptr = std::shared_ptr<Interaction>;

	// Non-member non-friend functions
	bool IsValid(const Interaction &);
	bool IsStatic(const Interaction &);
	bool HasTensor(const Interaction &);
	bool HasTrajectory(const Interaction &);
	InteractionType Type(const Interaction &);

	// Non-member non-friend functions for time-dependent fields
	arma::vec FieldTimeDependenceLinearPolarization(const arma::vec &, double, double, double);
	arma::vec FieldTimeDependenceCircularPolarization(const arma::vec &, double, double, double, const arma::vec &, bool);
	arma::vec FieldTimeDependenceBroadband(const arma::vec &, double, arma::mat, arma::mat, arma::mat, arma::mat, arma::mat, bool, int);
	arma::vec FieldTimeDependenceOUGeneral(const arma::vec &, arma::vec &, double, double, double, std::mt19937 &);

	// Non-member non-friend functions for ActionTarget validation
	bool CheckActionVectorInteractionField(const arma::vec &);
	bool CheckActionScalarInteractionPrefactor(const double &);

	//semi classical distributions
    void FreelyJointedPolymerBL(std::vector<double>&, std::vector<SCHyperfineField>&, double&, int); //length of the nuclear spin vector
	SCDistributionF FreelyJointedPolymerD(double,double); //Distribution of the nuclear spin vector
}

#endif
