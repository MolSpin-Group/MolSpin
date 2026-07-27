/////////////////////////////////////////////////////////////////////////
// SpinSystem class (SpinAPI Module)
// ------------------
// Collection of spins, interactions, and other characteristics of a
// single physical system.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_SpinAPI_SpinSystem
#define MOD_SpinAPI_SpinSystem

#include <memory>
#include <map>
#include <vector>
#include <iostream>
#include "SpinAPIDefines.h"
#include "SpinAPIfwd.h"
#include "ActionTarget.h"
#include "MSDParserfwd.h"

namespace SpinAPI
{
	class SpinSystem
	{
	private:
		// Implementation
		std::string name;
		std::vector<spin_ptr> spins;
		std::vector<interaction_ptr> interactions;
		std::vector<transition_ptr> transitions;
		std::vector<operator_ptr> operators;
		std::vector<pulse_ptr> pulses;
		std::vector<PulseSequence_ptr> pulses_seq;
		std::vector<state_ptr> states;
		std::vector<subsystem_ptr> subsystems;
		std::shared_ptr<MSDParser::ObjectParser> properties;

	public:
		// Constructors / Destructors
		explicit SpinSystem(std::string);		 // Normal constructor
		SpinSystem(const SpinSystem &) = delete; // Copy-constructor
		~SpinSystem();							 // Destructor

		// Operators
		const SpinSystem &operator=(const SpinSystem &) = delete; // Copy-assignment

		// Iterator methods
		// TODO: Return const_iterator-to-const
		decltype(spins.cbegin()) spins_cbegin() const { return this->spins.cbegin(); };
		decltype(spins.cend()) spins_cend() const { return this->spins.cend(); };
		decltype(interactions.cbegin()) interactions_cbegin() const { return this->interactions.cbegin(); };
		decltype(interactions.cend()) interactions_cend() const { return this->interactions.cend(); };
		decltype(states.cbegin()) states_cbegin() const { return this->states.cbegin(); };
		decltype(states.cend()) states_cend() const { return this->states.cend(); };
		decltype(transitions.cbegin()) transitions_cbegin() const { return this->transitions.cbegin(); };
		decltype(transitions.cend()) transitions_cend() const { return this->transitions.cend(); };
		decltype(operators.cbegin()) operators_cbegin() const { return this->operators.cbegin(); };
		decltype(operators.cend()) operators_cend() const { return this->operators.cend(); };
		decltype(pulses.cbegin()) pulses_cbegin() const { return this->pulses.cbegin(); };
		decltype(pulses.cend()) pulses_cend() const { return this->pulses.cend(); };
		decltype(pulses_seq.cbegin()) pulses_seq_cbegin() const {return this->pulses_seq.cbegin();};
		decltype(pulses_seq.cend()) pulses_seq_cend() const {return this->pulses_seq.cend();};

		// Object collections
		std::vector<spin_ptr> Spins() const;			   // Returns a copy of the collection of spins
		std::vector<interaction_ptr> Interactions() const; // Returns a copy of the collection of interactions
		std::vector<transition_ptr> Transitions() const;   // Returns a copy of the collection of transitions
		std::vector<operator_ptr> Operators() const;	   // Returns a copy of the collection of operators
		std::vector<pulse_ptr> Pulses() const;			   // Returns a copy of the collection of pulses
		std::vector<state_ptr> States() const;			   // Returns a copy of the collection of states
		std::vector<subsystem_ptr> SubSystems();		   // Returns a copy of the collection of subsystems
		std::vector<PulseSequence_ptr> PulseSequences();    // Returns a copy of the collection of PulseSequences;

		// Objects by name
		// TODO: Return pointer-to-const
		spin_ptr spins_find(const std::string &) const;				  // Returns the spin with the given name
		interaction_ptr interactions_find(const std::string &) const; // Returns the interaction with the given name
		state_ptr states_find(const std::string &) const;			  // Returns the spin state with the given name
		transition_ptr transitions_find(const std::string &) const;	  // Returns the transition with the given name
		operator_ptr operators_find(const std::string &) const;		  // Returns the operator with the given name
		pulse_ptr pulses_find(const std::string &) const;			  // Returns the pulse with the given name
		PulseSequence_ptr pulses_seq_find(const std::string &) const; // ''

		// Collection sizes
		unsigned int spins_size() const;		// Returns the number of spins in the system
		unsigned int interactions_size() const; // Returns the number of interactions in the system
		unsigned int states_size() const;		// Returns the number of spin states defined in the system
		unsigned int transitions_size() const;	// Returns the number of transitions in the system
		unsigned int operators_size() const;	// Returns the number of operators in the system
		unsigned int pulses_size() const;		// Returns the number of pulsess in the system
		unsigned int pulse_seq_size() const; 

		// Methods to check if an object is contained in the collections of the SpinSystem
		bool Contains(const spin_ptr &) const;
		bool Contains(const interaction_ptr &) const;
		bool Contains(const transition_ptr &) const;
		bool Contains(const operator_ptr &) const;
		bool Contains(const pulse_ptr &) const;
		bool Contains(const state_ptr &) const;
		bool Contains(const subsystem_ptr& ) const;
		bool Contains(const PulseSequence_ptr& ) const;

		// Add methods
		bool Add(const spin_ptr &);
		bool Add(const interaction_ptr &);
		bool Add(const transition_ptr &);
		bool Add(const operator_ptr &);
		bool Add(const pulse_ptr &);
		bool Add(const state_ptr &);
		bool Add(const subsystem_ptr& );
		bool Add(const PulseSequence_ptr& );

		// Subspace set methods
		bool IsComplete(const std::vector<spin_ptr> &) const; // Checks whether the spins comprise a complete subspace with respect to interactions and transitions (not all states)
		bool CompleteSet(std::vector<spin_ptr> &) const;	  // Extends the spin set (if necessary) such that the spins comprise a complete subspace with respect to interactions and transitions (not all states)

		// Validation methods
		std::vector<interaction_ptr> ValidateInteractions();														// Fills the interactions with spin_ptrs and checks if the interactions are valid
		std::vector<transition_ptr> ValidateTransitions(const std::vector<std::shared_ptr<SpinAPI::SpinSystem>> &); // Loads states into the transition objects and checks if the transitions are valid
		std::vector<operator_ptr> ValidateOperators(const std::vector<std::shared_ptr<SpinAPI::SpinSystem>> &);		// Validates operators
		std::vector<pulse_ptr> ValidatePulses();																	// Validates pulses
		std::vector<PulseSequence_ptr> ValidatePulseSequences(); 													// Validates Pulse sequences
		std::vector<state_ptr> ValidateStates();																	// Attempts to parse all the states (requires all spins to be read into this->spins first). Returns list of states that failed validation/parsing.
		std::vector<subsystem_ptr> ValidateSubSystems(); 															// Validates Subsystems;

		// Properties
			bool SetProperties(const std::shared_ptr<MSDParser::ObjectParser> &);
			std::vector<state_ptr> InitialState() const;
			StateFrame InitialStateFrame() const;
			InitialStateCoherenceMode InitialStateCoherences() const;
			double Temperature();		   // Obtain temperature for temperature weighted density matrix
			std::vector<double> Weights(); // Obtain weights for specifically weighted density matrix
			std::vector<std::string> ThermalHamiltonianList();

		// Other public methods
		void GetActionTargets(std::map<std::string, RunSection::ActionScalar> &, std::map<std::string, RunSection::ActionVector> &) const;
		std::string Name() const { return this->name; };
		void Print(bool _printFullState = false);
		std::vector<state_ptr> GetAllStates() { return this->states; };
		std::shared_ptr<MSDParser::ObjectParser> GetProperties() const {return this->properties;};

		template <typename t>
		bool RemoveFailedObjects(std::vector<t>& failed, std::vector<t>& total)
		{
			std::vector<t> tmpvector = total;
			int removed = 0;
			for(unsigned int i = 0; i < failed.size(); i++)
			{
				auto it = std::find(total.begin(), total.end(), failed[i-removed]);
				total.erase(it);
				removed += 1;
			}
			return true;
		}
		 
	};

	using system_ptr = std::shared_ptr<SpinAPI::SpinSystem>;

	// Subspace management functions (non-member non-friend)
	std::vector<std::vector<spin_ptr>> CompleteSubspaces(const SpinSystem &);		 // Returns collection of complete subspaces
	bool IsCompleteSubspace(const std::vector<spin_ptr> &, const SpinSystem &);		 // Checks whether the given subspace is complete with respect to the given SpinSystem (all interactions and transitions)
	bool IsCompleteSubspace(const std::vector<spin_ptr> &, const interaction_ptr &); // Checks whether the given subspace is complete with respect to the given interaction
	bool IsCompleteSubspace(const std::vector<spin_ptr> &, const state_ptr &);		 // Checks whether the given subspace is complete with respect to the given state (i.e. no outside entanglements)
	bool IsCompleteSubspace(const std::vector<spin_ptr> &, const transition_ptr &);	 // Checks whether the given subspace is complete with respect to the given transition
	bool CompleteSubspace(std::vector<spin_ptr> &, const SpinSystem &);				 // Extends the subspace such that it is complete with respect to a SpinSystem. Returns true if the subspace was extend
	bool CompleteSubspace(std::vector<spin_ptr> &, const interaction_ptr &);		 // Extends the subspace such that it is complete with respect to an Interactions. Returns true if the subspace was extend
	bool CompleteSubspace(std::vector<spin_ptr> &, const state_ptr &);				 // Extends the subspace such that it is complete with respect to a State. Returns true if the subspace was extend
	bool CompleteSubspace(std::vector<spin_ptr> &, const transition_ptr &);			 // Extends the subspace such that it is complete with respect to a Transition. Returns true if the subspace was extend
}

#endif
