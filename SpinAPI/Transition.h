/////////////////////////////////////////////////////////////////////////
// Transition class (SpinAPI Module)
// ------------------
// Describes reaction operators.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_SpinAPI_Transition
#define MOD_SpinAPI_Transition

#include <memory>
#include "Trajectory.h"
#include "SpinSystem.h"
#include "MSDParserfwd.h"

namespace SpinAPI
{
	class Transition
	{
	private:
		// Implementation details
		std::shared_ptr<SpinSystem> system;
		std::shared_ptr<SpinSystem> target;
		std::shared_ptr<MSDParser::ObjectParser> properties;
		double rate;
		TransitionType type;
		state_ptr sourcestate;
		state_ptr targetstate;
		ReactionOperatorType reactionOperators;
		bool isValid;

		// Trajectory parameters
		Trajectory trajectory;
		bool trjHasTime;
		bool trjHasRate;
		bool trjRateIsLifetime; // Whether lifetime was specified instead of rate (trjRate will then point to the lifetime column)
		unsigned int trjTime;
		unsigned int trjRate;

		// Private methods to create ActionTargets
		std::vector<RunSection::NamedActionScalar> CreateActionScalars(const std::string &);

		//flags for pulsing
		bool Pulsed;
		bool Active;
		double active_time;

	public:
		// Constructors / Destructors
		Transition(std::string, std::string, std::shared_ptr<SpinSystem>); // Normal constructor
		Transition(const Transition &);									   // Copy-constructor
		~Transition();													   // Destructor

		// Operators
		const Transition &operator=(const Transition &); // Copy-assignment

		// Name and validation
		std::string Name() const;
		bool Validate(const std::vector<std::shared_ptr<SpinAPI::SpinSystem>> &_systems); // A validation method that gets the state_ptrs from the systems
		bool IsValid() const;
		bool HasTrajectory() const;

		void SetValid(bool valid)
		{
			isValid = valid;
		}

		// Public property methods
		double Rate() const;
		TransitionType Type() const;
		std::shared_ptr<SpinSystem> System() const;
		std::shared_ptr<SpinSystem> Target() const;
		state_ptr SourceState() const;
		state_ptr TargetState() const;
		ReactionOperatorType GetReactionOperatorType() const;
		unsigned int TrajectoryLength() const; // Number of steps in the trajectory (0 means no trajectory)
		bool SetTrajectoryStep(unsigned int);  // Set the rate based on a trajectory
		bool SetTime(double);				   // Set the rate based on a trajectory

		// Allow access to custom properties to be used for custom tasks
		std::shared_ptr<const MSDParser::ObjectParser> Properties() const;

		// Spin subspace methods
		bool IsComplete(const std::vector<spin_ptr> &) const;		// Checks whether any spins in the source state are coupled/entangled to any spins outside the source state
		bool IsTargetComplete(const std::vector<spin_ptr> &) const; // Checks whether any spins in the target state are coupled/entangled to any spins outside the target state
		bool CompleteSet(std::vector<spin_ptr> &) const;			// Extends the set such that all spins inside the source state can only be coupled/entangled to other spins inside the source state
		bool CompleteTargetSet(std::vector<spin_ptr> &) const;		// Extends the set such that all spins inside the target state can only be coupled/entangled to other spins inside the target state

		// Public method for creating ActionTargets
		void GetActionTargets(std::vector<RunSection::NamedActionScalar> &, std::vector<RunSection::NamedActionVector> &, const std::string &);

		bool IsActive()
		{
			if (this->Pulsed == true)
			{
				bool act = this->Active;
				this->Active = false; //this essentially ensures the interaction/transition is only active during the pulse time
				return act;
			}
			return true;
		}

		void SetPulsed(bool pulsed)
		{
			this->Pulsed = pulsed;
		}

		bool GetPulsed() const
		{
			return this->Pulsed;
		}

		void SetActive(bool act)
		{
			this->Active = act;
		}

		bool HasTD() const
		{
			if (this->GetPulsed())
			{
				return true;
			}
			return this->HasTrajectory();
		}

		double ActiveTime() const
		{
			return this->active_time;
		}

	};

	// Define alias for transition-pointers
	using transition_ptr = std::shared_ptr<Transition>;

	// Non-member non-friend functions
	bool IsValid(const Transition &);
	bool IsStatic(const Transition &);
	bool HasTrajectory(const Transition &);

	// Non-member non-friend functions for ActionTarget validation
	bool CheckActionScalarTransitionRate(const double &);
}

#endif
