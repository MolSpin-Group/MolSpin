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
#include "Spin.h"
#include "Interaction.h"
#include "Transition.h"
#include "Operator.h"
#include "Pulse.h"
#include "PulseSequence.h"
#include "State.h"
#include "SubSystem.h"
#include "ObjectParser.h"
#include "SpinSystem.h"
#include <algorithm>
#include <cctype>

namespace SpinAPI
{
	// -----------------------------------------------------
	// SpinSystem Constructors and Destructor
	// -----------------------------------------------------
	SpinSystem::SpinSystem(std::string _name) : name(_name), spins(), interactions(), transitions(), operators(), states(), properties(nullptr)
	{
	}

	SpinSystem::~SpinSystem()
	{
	}
	// -----------------------------------------------------
	// Public object collection methods
	// -----------------------------------------------------
	// Returns a list containing all the spins in the SpinSystem
	std::vector<spin_ptr> SpinSystem::Spins() const
	{
		return this->spins;
	}

	// Returns a list containing all the interactions in the SpinSystem
	std::vector<interaction_ptr> SpinSystem::Interactions() const
	{
		return this->interactions;
	}

	// Returns a list containing all the transitions in the SpinSystem
	std::vector<transition_ptr> SpinSystem::Transitions() const
	{
		return this->transitions;
	}

	// Returns a list containing all the operators in the SpinSystem
	std::vector<operator_ptr> SpinSystem::Operators() const
	{
		return this->operators;
	}

	// Returns a list containing all the pulses in the SpinSystem
	std::vector<pulse_ptr> SpinSystem::Pulses() const
	{
		return this->pulses;
	}

	// Returns a list containing all the states in the SpinSystem
	std::vector<state_ptr> SpinSystem::States() const
	{
		return this->states;
	}

	std::vector<subsystem_ptr> SpinSystem::SubSystems()
	{
		return this->subsystems;
	}

    std::vector<PulseSequence_ptr> SpinSystem::PulseSequences()
    {
        return this->pulses_seq;
    }
    // -----------------------------------------------------
    // Public methods to find objects by name
    // -----------------------------------------------------
    // Returns the Spin object with the given name
    spin_ptr SpinSystem::spins_find(const std::string &_name) const
	{
		for (auto i = this->spins.cbegin(); i != this->spins.cend(); i++)
			if ((*i)->Name().compare(_name) == 0)
				return (*i);

		return nullptr;
	}

	// Returns the Interaction object with the given name
	interaction_ptr SpinSystem::interactions_find(const std::string &_name) const
	{
		for (auto i = this->interactions.cbegin(); i != this->interactions.cend(); i++)
			if ((*i)->Name().compare(_name) == 0)
				return (*i);

		return nullptr;
	}

	// Returns the Transition object with the given name
	transition_ptr SpinSystem::transitions_find(const std::string &_name) const
	{
		for (auto i = this->transitions.cbegin(); i != this->transitions.cend(); i++)
			if ((*i)->Name().compare(_name) == 0)
				return (*i);

		return nullptr;
	}

	// Returns the Operator object with the given name
	operator_ptr SpinSystem::operators_find(const std::string &_name) const
	{
		for (auto i = this->operators.cbegin(); i != this->operators.cend(); i++)
			if ((*i)->Name().compare(_name) == 0)
				return (*i);

		return nullptr;
	}

	// Returns the Pulse object with the given name
	pulse_ptr SpinSystem::pulses_find(const std::string &_name) const
	{
		for (auto i = this->pulses.cbegin(); i != this->pulses.cend(); i++)
			if ((*i)->Name().compare(_name) == 0)
				return (*i);

		return nullptr;
	}

	PulseSequence_ptr SpinSystem::pulses_seq_find(const std::string &_name) const
	{
		for (auto i = this->pulses_seq.cbegin(); i != this->pulses_seq.cend(); i++)
			if ((*i)->Name().compare(_name) == 0)
				return (*i);

		return nullptr;
	}

	// Returns the state object with given name
	state_ptr SpinSystem::states_find(const std::string &_name) const
	{
		for (auto i = this->states.cbegin(); i != this->states.cend(); i++)
			if ((*i)->Name().compare(_name) == 0)
				return (*i);

		return nullptr;
	}
	// -----------------------------------------------------
	// Public collection size methods
	// -----------------------------------------------------
	// Return the number of Spin objects within the SpinSystem
	unsigned int SpinSystem::spins_size() const
	{
		return this->spins.size();
	}

	// Returns the number of Interaction objects in the SpinSystem
	unsigned int SpinSystem::interactions_size() const
	{
		return this->interactions.size();
	}

	// Returns the number of Transition objects in the SpinSystem
	unsigned int SpinSystem::transitions_size() const
	{
		return this->transitions.size();
	}

	// Returns the number of Operator objects in the SpinSystem
	unsigned int SpinSystem::operators_size() const
	{
		return this->operators.size();
	}

	// Returns the number of Pulse objects in the SpinSystem
	unsigned int SpinSystem::pulses_size() const
	{
		return this->pulses.size();
	}

	unsigned int SpinSystem::pulse_seq_size() const
	{
		return this->pulses_seq.size();
	}

	// Returns the number of State objects defined for the SpinSystem
	unsigned int SpinSystem::states_size() const
	{
		return this->states.size();
	}
	// -----------------------------------------------------
	// Public methods to check whether an object is
	// contained in the collections of the SpinSystem
	// -----------------------------------------------------
	// Checks whether a spin object is contained by the SpinSystem
	bool SpinSystem::Contains(const spin_ptr &_spin) const
	{
		for (auto i = this->spins.cbegin(); i != this->spins.cend(); i++)
			if ((*i) == _spin)
				return true;
		return false;
	}

	// Checks whether a interaction object is contained by the SpinSystem
	bool SpinSystem::Contains(const interaction_ptr &_interaction) const
	{
		for (auto i = this->interactions.cbegin(); i != this->interactions.cend(); i++)
			if ((*i) == _interaction)
				return true;
		return false;
	}

	// Checks whether a transition object is contained by the SpinSystem
	bool SpinSystem::Contains(const transition_ptr &_transition) const
	{
		for (auto i = this->transitions.cbegin(); i != this->transitions.cend(); i++)
			if ((*i) == _transition)
				return true;
		return false;
	}

	// Checks whether an operator object is contained by the SpinSystem
	bool SpinSystem::Contains(const operator_ptr &_operator) const
	{
		for (auto i = this->operators.cbegin(); i != this->operators.cend(); i++)
			if ((*i) == _operator)
				return true;
		return false;
	}

	// Checks whether a pulse object is contained by the SpinSystem
	bool SpinSystem::Contains(const pulse_ptr &_pulse) const
	{
		for (auto i = this->pulses.cbegin(); i != this->pulses.cend(); i++)
			if ((*i) == _pulse)
				return true;
		return false;
	}

	bool SpinSystem::Contains(const PulseSequence_ptr & ptr) const
	{
		if (this->pulses_seq_find(ptr->Name()) == nullptr)
			return false;
		return true;
	}

	// Checks whether a state object is contained by the SpinSystem

	bool SpinSystem::Contains(const state_ptr &_state) const
	{
		for (auto i = this->states.cbegin(); i != this->states.cend(); i++)
			if ((*i) == _state)
				return true;
		return false;
	}
	// Checks whether a SubSystem object is contained by the SpinSystem
	bool SpinSystem::Contains(const subsystem_ptr &_sub) const
	{
		for (auto i = this->subsystems.cbegin(); i != subsystems.cend(); i++)
		{
			if ((*i) == _sub)
			{
				return true;
			}
		}
		return false;
	}
	// -----------------------------------------------------
	// Public methods to add objects to the collections
	// -----------------------------------------------------
	// Add a Spin object to the SpinSystem
	bool SpinSystem::Add(const spin_ptr &_spin)
	{
		if (this->Contains(_spin))
			return false;

		if (!_spin->IsValid())
		{
			std::cerr << "Warning: Failed to add spin \"" << _spin->Name() << "\" to SpinSystem " << this->Name() << ". Invalid spin object!" << std::endl;
			return false;
		}

		this->spins.push_back(_spin);
		return true;
	}

	// Adds an Interaction object to the SpinSystem. The Interaction objects are not valid until SpinSystem::ValidateInteractions is called.
	bool SpinSystem::Add(const interaction_ptr &_interaction)
	{
		if (this->Contains(_interaction))
			return false;

		this->interactions.push_back(_interaction);
		return true;
	}

	// Adds a Transition object to the SpinSystem
	bool SpinSystem::Add(const transition_ptr &_transition)
	{
		if (this->Contains(_transition))
			return false;

		this->transitions.push_back(_transition);
		return true;
	}

	// Adds an Operator object to the SpinSystem
	bool SpinSystem::Add(const operator_ptr &_operator)
	{
		if (this->Contains(_operator))
			return false;

		this->operators.push_back(_operator);
		return true;
	}

	
	// Adds an Pulse object to the SpinSystem. The Interaction objects are not valid until SpinSystem::ValidateInteractions is called.
	bool SpinSystem::Add(const pulse_ptr &_pulse)
	{
		if (this->Contains(_pulse))
			return false;

		this->pulses.push_back(_pulse);
		return true;
	}

	bool SpinSystem::Add(const PulseSequence_ptr &pulseseq)
	{
		if(this->Contains(pulseseq))
			return false;
		this->pulses_seq.push_back(pulseseq);
		return true;
	}

	// Define a new state on the SpinSystem. Note that it will not be valid until SpinSystem::ValidateStates has been called.
	bool SpinSystem::Add(const state_ptr &_state)
	{
		if (this->Contains(_state))
			return false;

		this->states.push_back(_state);
		return true;
	}

	// Adds a SubSystem to the SpinSystem. The SubSystem will not be valid until SpinSystem::ValidateSubSystem has been called
	bool SpinSystem::Add(const subsystem_ptr &_sub)
	{
		if (this->Contains(_sub))
			return false;
		this->subsystems.push_back(_sub);
		return true;
	}
	// -----------------------------------------------------
	// Subspace management methods (SpinSystem class only)
	// See non-member non-friend functions below.
	// -----------------------------------------------------
	// Checks whether the spins comprise a complete subspace with respect to interactions and transitions (not all states)
	bool SpinSystem::IsComplete(const std::vector<spin_ptr> &_subspace) const
	{
		bool isComplete = true;

		// Check interactions
		for (auto i = this->interactions.cbegin(); i != this->interactions.cend(); i++)
			isComplete &= (*i)->IsComplete(_subspace);

		// Check transitions
		for (auto i = this->transitions.cbegin(); i != this->transitions.cend(); i++)
			isComplete &= (*i)->IsComplete(_subspace);

		return isComplete;
	}

	// Extends the spin set (if necessary) such that the spins comprise a complete subspace with respect to interactions and transitions (not all states)
	// Returns true if the spin set was extended
	bool SpinSystem::CompleteSet(std::vector<spin_ptr> &_subspace) const
	{
		bool hasChanged = false;

		// Extend set by interactions
		for (auto i = this->interactions.cbegin(); i != this->interactions.cend(); i++)
			hasChanged |= (*i)->CompleteSet(_subspace);

		// Extend set by transitions
		for (auto i = this->transitions.cbegin(); i != this->transitions.cend(); i++)
			hasChanged |= (*i)->CompleteSet(_subspace);

		// Calling "CompleteSet" once on a collection of objects may not be enough, so we call it repeatedly until no new objects are added
		bool notSelfConsistent = true;
		while (notSelfConsistent)
		{
			notSelfConsistent = false;
			// Extend set by interactions
			for (auto i = this->interactions.cbegin(); i != this->interactions.cend(); i++)
				notSelfConsistent |= (*i)->CompleteSet(_subspace);

			// Extend set by transitions
			for (auto i = this->transitions.cbegin(); i != this->transitions.cend(); i++)
				notSelfConsistent |= (*i)->CompleteSet(_subspace);
		}

		return hasChanged;
	}
	// -----------------------------------------------------
	// Public Validation methods
	// -----------------------------------------------------
	// Prepares interaction_ptr objects and checks whether they are valid
	std::vector<interaction_ptr> SpinSystem::ValidateInteractions()
	{
		std::vector<interaction_ptr> failedInteractions;
		for (auto i = this->interactions.cbegin(); i != this->interactions.cend(); i++)
		{
			if ((*i)->ParseSpinGroups(this->spins))
			{
				if (!(*i)->IsValid())
					failedInteractions.push_back(*i);
			}
			else
			{
				failedInteractions.push_back(*i);
			}
		}

		// TODO: Remove range failedInteractions from this->interactions
		RemoveFailedObjects<interaction_ptr>(failedInteractions,this->interactions);
		return failedInteractions;
	}

	// Prepares transition_ptr objects and checks whether they are valid
	std::vector<transition_ptr> SpinSystem::ValidateTransitions(const std::vector<std::shared_ptr<SpinAPI::SpinSystem>> &_systems)
	{
		std::vector<transition_ptr> failedTransitions;
		for (auto i = this->transitions.cbegin(); i != this->transitions.cend(); i++)
		{
			if (!(*i)->Validate(_systems))
			{
				failedTransitions.push_back(*i);
			}
		}

		// TODO: Remove range failedTransitions from this->transitions
		RemoveFailedObjects<transition_ptr>(failedTransitions, this->transitions);
		return failedTransitions;
	}

	// Prepares operator_ptr objects and checks whether they are valid
	std::vector<operator_ptr> SpinSystem::ValidateOperators(const std::vector<std::shared_ptr<SpinAPI::SpinSystem>> &_systems)
	{
		std::vector<operator_ptr> failedOperators;
		for (auto i = this->operators.cbegin(); i != this->operators.cend(); i++)
		{
			if (!(*i)->Validate(_systems))
			{
				failedOperators.push_back(*i);
			}
		}

		// TODO: Remove range failedOperators from this->operators
		RemoveFailedObjects<operator_ptr>(failedOperators, this->operators);
		return failedOperators;
	}

	// Prepares pulse_ptr objects and checks whether they are valid
	std::vector<pulse_ptr> SpinSystem::ValidatePulses()
	{
		std::vector<pulse_ptr> failedPulses;
		for (auto i = this->pulses.cbegin(); i != this->pulses.cend(); i++)
		{
			if ((*i)->ParseSpinGroups(this->spins))
			{
				if (!(*i)->IsValid())
					failedPulses.push_back(*i);
			}
			else
			{
				failedPulses.push_back(*i);
			}
		}
		
		RemoveFailedObjects<pulse_ptr>(failedPulses, this->pulses);
		return failedPulses;
	}

	std::vector<PulseSequence_ptr> SpinSystem::ValidatePulseSequences()
	{
		std::vector<PulseSequence_ptr>failedPulseSequences;
		for(auto i = this->pulses_seq.cbegin(); i != this->pulses_seq.cend(); i++)
		{
			if((*i)->ParsePulseSequence(this->pulses,this->interactions,this->transitions))
			{
				if(!(*i)->IsValid())
					failedPulseSequences.push_back(*i);
			}
			else
			{
				failedPulseSequences.push_back(*i);
			}
		}
		RemoveFailedObjects<PulseSequence_ptr>(failedPulseSequences, this->pulses_seq);
		return failedPulseSequences;
	}

	// Method that loads the state objects and checks whether they are valid
	std::vector<state_ptr> SpinSystem::ValidateStates()
	{
		std::vector<state_ptr> failedStates;
		for (auto i = this->states.cbegin(); i != this->states.cend(); i++)
		{
			if (!((*i)->ParseFromSystem(*this)))
			{
				failedStates.push_back(*i);
			}
		}

		// TODO: Remove range failedStates from this->states
		RemoveFailedObjects<state_ptr>(failedStates, this->states);
		return failedStates;
	}

	std::vector<subsystem_ptr> SpinSystem::ValidateSubSystems()
	{
		std::vector<subsystem_ptr> failedSubSystems;
		for (auto i = this->subsystems.cbegin(); i != this->subsystems.cend(); i++)
		{
			if (!((*i)->Validate()))
			{
				failedSubSystems.push_back(*i);
			}
		}

		RemoveFailedObjects<subsystem_ptr>(failedSubSystems, this->subsystems);
		return failedSubSystems;
	}
	// -----------------------------------------------------
	// Properties
	// -----------------------------------------------------
	// Sets a properties object for the SpinSystem. Only possible if none was assigned previously.
	bool SpinSystem::SetProperties(const std::shared_ptr<MSDParser::ObjectParser> &_properties)
	{
		if (this->properties != nullptr)
			return false;

		this->properties = _properties;
		return true;
	}

	// Returns a list of initial states (if any was specified)
	// Initial states will be specified as a comma-separated list.
	// Normally there would be only one initial state, but it should be possible to combine state objects (e.g. to create overall "Triplet" from T0, T+ and T- states)
	std::vector<state_ptr> SpinSystem::InitialState() const
	{
		// Create a vector object to hold all the states specified (if any)
		std::vector<state_ptr> iniStates;

		// Check whether a properties object was given
		if (this->properties != nullptr)
		{
			// Check whether an initial state was specified
			std::string str;
			if (this->properties->Get("initialstate", str))
			{
				// Get all the specified initial states separated by comma
				std::istringstream stream(str);
				for (std::string s; std::getline(stream, s, ',');)
				{
					if (s.compare("Thermal") == 0 || s.compare("thermal") == 0)
					{
						iniStates.push_back(nullptr);
					}
					else
					{
						// Find the state with the given name
						for (auto i = this->states.cbegin(); i != this->states.cend(); i++)
						{
							if ((*i)->Name().compare(s) == 0)
								iniStates.push_back(*i);
						}
					}
				}
			}
		}

		return iniStates;
	}

	StateFrame SpinSystem::InitialStateFrame() const
	{
		if (this->properties == nullptr)
			return StateFrame::Fixed;

		std::string str;
		if (!this->properties->Get("frame", str) && !this->properties->Get("stateframe", str))
			return StateFrame::Fixed;

		// Accept a few short aliases because this keyword appears in user MSD
		// files. The returned enum is used by powder tasks when they decide
		// whether the initial density matrix follows the molecular frame.
		std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		if (str == "fixed" || str == "lab" || str == "laboratory" || str == "default")
			return StateFrame::Fixed;
		if (str == "molecular" || str == "mol" || str == "rotating")
			return StateFrame::Molecular;
		if (str == "eigen" || str == "thermal")
			return StateFrame::Eigen;

		return StateFrame::Fixed;
	}

	InitialStateCoherenceMode SpinSystem::InitialStateCoherences() const
	{
		if (this->properties == nullptr)
			return InitialStateCoherenceMode::Keep;

		std::string str;
		if (!this->properties->Get("initialstatecoherences", str) &&
			!this->properties->Get("initialcoherences", str) &&
			!this->properties->Get("coherences", str))
		{
			return InitialStateCoherenceMode::Keep;
		}

		// The "dephase" mode is basis dependent. The task supplies the
		// orientation-specific Hamiltonian used to define that eigenbasis.
		std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		if (str == "keep" || str == "full" || str == "coherent" || str == "default")
			return InitialStateCoherenceMode::Keep;
		if (str == "dephase" || str == "discard" || str == "diagonal" || str == "populations" ||
			str == "eigen" || str == "eigenbasis" || str == "eigenpopulations")
		{
			return InitialStateCoherenceMode::DephaseEigenbasis;
		}

		std::cerr << "Warning: Unknown initial-state coherences mode \"" << str << "\". Keeping coherences." << std::endl;
		return InitialStateCoherenceMode::Keep;
	}

	// Obtain the temperature for a temperature weighted density matrix
	double SpinSystem::Temperature()
	{
		double temperature;
		if (!this->properties->Get("temperature", temperature))
		{
			std::cerr << "Warning: Failed to find temperature value. We set the temperature to 300 K." << std::endl;
			temperature = 300;
		}

		return temperature;
	}

	std::vector<std::string> SpinSystem::ThermalHamiltonianList()
	{
		std::vector<std::string> thermalhamiltonian_list;
		this->properties->GetList("thermalhamiltonian", thermalhamiltonian_list, ',');

		return thermalhamiltonian_list;
	}

	// Obtain the weights for a specifically weighted density matrix
	std::vector<double> SpinSystem::Weights()
	{
		std::vector<double> weights;
		this->properties->GetList("weights", weights);

		return weights;
	}

	// -----------------------------------------------------
	// Other public methods
	// -----------------------------------------------------
	// Creates ActionTarget objects for the RunSection
	void SpinSystem::GetActionTargets(std::map<std::string, RunSection::ActionScalar> &_scalars, std::map<std::string, RunSection::ActionVector> &_vectors) const
	{
		// Create temporary containers to collect all the ActionTargets
		std::vector<RunSection::NamedActionScalar> tmpVecScalars;
		std::vector<RunSection::NamedActionVector> tmpVecVectors;

		// Get ActionTargets from all the Spin objects, adding the name of the SpinSystem
		for (auto i = this->spins.cbegin(); i != this->spins.cend(); i++)
			(*i)->GetActionTargets(tmpVecScalars, tmpVecVectors, this->Name());

		// Get ActionTargets from all the Interaction objects, adding the name of the SpinSystem
		for (auto i = this->interactions.cbegin(); i != this->interactions.cend(); i++)
			(*i)->GetActionTargets(tmpVecScalars, tmpVecVectors, this->Name());

		// Get ActionTargets from all the Transition objects, adding the name of the SpinSystem
		for (auto i = this->transitions.cbegin(); i != this->transitions.cend(); i++)
			(*i)->GetActionTargets(tmpVecScalars, tmpVecVectors, this->Name());

		// Get ActionTargets from all the Pulse objects, adding the name of the SpinSystem
		for (auto i = this->pulses.cbegin(); i != this->pulses.cend(); i++)
			(*i)->GetActionTargets(tmpVecScalars, tmpVecVectors, this->Name());

		// Get ActionTargets from all state objects, adding the name of the SpinSystem
		for (auto i = this->states.cbegin(); i != this->states.cend(); i++)
			(*i)->GetActionTargets(tmpVecScalars, tmpVecVectors, this->Name());

		// Insert all the ActionScalars in the associated container
		for (auto i = tmpVecScalars.cbegin(); i != tmpVecScalars.cend(); i++)
		{
			auto j = _scalars.insert((*i));
			if (!j.second)
				std::cout << "Could not create ActionScalar named \"" << j.first->first << "\" because it already exists!" << std::endl;
		}

		// Insert all the ActionVectors in the associated container
		for (auto i = tmpVecVectors.cbegin(); i != tmpVecVectors.cend(); i++)
		{
			auto j = _vectors.insert((*i));
			if (!j.second)
				std::cout << "Could not create ActionVector named \"" << j.first->first << "\" because it already exists!" << std::endl;
		}
	}

	// Prints information about the SpinSystem
	void SpinSystem::Print(bool _printFullState)
	{
		std::cout << "###### SpinSystem \"" << this->Name() << "\":" << std::endl;

		// Information about the spins
		std::cout << "# Spins (" << this->spins.size() << "):" << std::endl;
		for (auto i = this->spins.cbegin(); i != this->spins.cend(); i++)
		{
			const Tensor &t = (*i)->GetTensor();
			std::cout << " - " << (*i)->Name() << " (spin: " << (*i)->S() << "/2, multiplicity: " << (*i)->Multiplicity() << ", type: ";
			if ((*i)->Type() == SpinType::Electron)
			{
				std::cout << "electron";
			}
			else if ((*i)->Type() == SpinType::Nucleus)
			{
				std::cout << "nucleus";
			}
			else if ((*i)->Type() == SpinType::NotSpecified)
			{
				std::cout << "not specified";
			}
			else
			{
				std::cout << "unknown";
			}
			std::cout << ", trajectory length: " << (*i)->TrajectoryLength() << ", isotropic: " << t.Isotropic() << ", anisotropic: " << t.Anisotropic()(0) << " ";
			std::cout << t.Anisotropic()(1) << " " << " " << t.Anisotropic()(2) << ", is valid: " << (*i)->IsValid() << ")" << std::endl;
		}

		// Information about the interactions
		std::cout << "\n# Interactions (" << this->interactions.size() << "):" << std::endl;
		for (auto i = this->interactions.cbegin(); i != this->interactions.cend(); i++)
		{
			std::cout << " - " << (*i)->Name() << " (type: ";
			if ((*i)->Type() == InteractionType::SingleSpin)
			{
				std::cout << "SingleSpin, field-type: ";
				if ((*i)->FieldType() == InteractionFieldType::Static)
				{
					std::cout << "Static";
				}
				else if ((*i)->FieldType() == InteractionFieldType::LinearPolarization)
				{
					std::cout << "LinearPolarization";
				}
				else if ((*i)->FieldType() == InteractionFieldType::CircularPolarization)
				{
					std::cout << "CircularPolarization";
				}
				else
				{
					std::cout << "Unknown";
				}
				std::cout << ", spins: " << (*i)->Group1().size();
			}
			else if ((*i)->Type() == InteractionType::DoubleSpin)
			{
				std::cout << "DoubleSpin, group1: " << (*i)->Group1().size() << ", group2: " << (*i)->Group2().size();
			}
			else
			{
				std::cout << "Unknown";
			}
			std::cout << ", prefactor: " << (*i)->Prefactor() << ", trajectory length: " << (*i)->TrajectoryLength();
			std::cout << ", is valid: " << (*i)->IsValid() << ")" << std::endl;
		}

		// Information about transitions
		std::cout << "\n# Transitions (" << this->transitions.size() << "):" << std::endl;
		for (auto i = this->transitions.cbegin(); i != this->transitions.cend(); i++)
		{
			state_ptr tmp = nullptr;
			std::cout << " - " << (*i)->Name() << " (rate: " << (*i)->Rate() << ", source: ";
			if ((tmp = (*i)->SourceState()) != nullptr)
			{
				std::cout << tmp->Name();
			}
			else
			{
				std::cout << "none";
			}
			std::cout << ", target: ";
			if ((tmp = (*i)->TargetState()) != nullptr)
			{
				std::cout << tmp->Name();
			}
			else
			{
				std::cout << "none";
			}
			std::cout << ", type: ";
			if ((*i)->Type() == TransitionType::Source)
			{
				std::cout << "source";
			}
			else if ((*i)->Type() == TransitionType::Sink)
			{
				std::cout << "sink";
			}
			else
			{
				std::cout << "unknown";
			}
			std::cout << ", trajectory length: " << (*i)->TrajectoryLength() << ", is valid: " << (*i)->IsValid() << ")" << std::endl;
		}

		// Information about operators
		std::cout << "\n# Operators (" << this->operators.size() << "):" << std::endl;
		for (auto i = this->operators.cbegin(); i != this->operators.cend(); i++)
		{
			std::cout << " - " << (*i)->Name() << " (type: ";
			if ((*i)->Type() == OperatorType::Unspecified)
			{
				std::cout << "Unspecified";
			}
			else if ((*i)->Type() == OperatorType::RelaxationLindblad)
			{
				std::cout << "RelaxationLindblad";
			}
			else if ((*i)->Type() == OperatorType::RelaxationDephasing)
			{
				std::cout << "RelaxationDephasing";
			}
			else if ((*i)->Type() == OperatorType::RelaxationLindbladDoubleSpin)
			{
				std::cout << "RelaxationLindbladDoubleSpin";
			}
			else if ((*i)->Type() == OperatorType::RelaxationRandomFields)
			{
				std::cout << "RelaxationRandomFields";
			}
			else if ((*i)->Type() == OperatorType::RelaxationT1)
			{
				std::cout << "RelaxationT1";
			}
			else if ((*i)->Type() == OperatorType::RelaxationT2)
			{
				std::cout << "RelaxationT2";
			}
			else if ((*i)->Type() == OperatorType::RelaxationPhenomenological)
			{
				std::cout << "RelaxationPhenomenological";
			}
			else
			{
				std::cout << "unknown";
			}
			std::cout << ", spins: " << (*i)->SpinCount();
			std::cout << ", frame: " << ((*i)->Frame() == RelaxationFrame::Lab ? "lab" : "molecular");
			std::cout << ", is valid: " << (*i)->IsValid() << ")" << std::endl;
		}

		// Information about pulses
		std::cout << "\n# Pulses (" << this->pulses.size() << "):" << std::endl;
		for (auto i = this->pulses.cbegin(); i != this->pulses.cend(); i++)
		{
			std::cout << " - " << (*i)->Name() << " (type: ";
			if ((*i)->Type() == PulseType::Unspecified)
			{
				std::cout << "Unspecified";
			}
			else if ((*i)->Type() == PulseType::InstantPulse)
			{
				std::cout << "InstantPulse";
			}
			else if ((*i)->Type() == PulseType::LongPulse)
			{
				std::cout << "LongPulse";
			}
			else if ((*i)->Type() == PulseType::MWPulse)
			{
				std::cout << "MWPulse";
			}
			else if ((*i)->Type() == PulseType::LongPulseStaticField)
			{
				std::cout << "LongPulseStaticField";
			}
			else if ((*i)->Type() == PulseType::ShapedPulse)
			{
				std::cout << "ShapedPulse";
			}
			else
			{
				std::cout << "unknown";
			}
			std::cout << ", is valid: " << (*i)->IsValid() << ")" << std::endl;
		}

		// Information about states
		std::cout << "\n# States (" << this->states.size() << "):" << std::endl;
		if (_printFullState)
		{
			for (auto i = this->states.cbegin(); i != this->states.cend(); i++)
			{
				std::cout << " - Printing state object \"" << (*i)->Name() << "\":" << std::endl;
				(*i)->Print(std::cout);
				std::cout << std::endl;
			}
		}
		else
		{
			for (auto i = this->states.cbegin(); i != this->states.cend(); i++)
			{
				std::cout << " - " << (*i)->Name() << " (is valid: " << (*i)->IsValid() << ")" << std::endl;
			}
		}
	}
	// -----------------------------------------------------
	// Subspace management functions (non-member non-friend)
	// -----------------------------------------------------
	// Returns collection of complete subspaces
	std::vector<std::vector<spin_ptr>> CompleteSubspaces(const SpinSystem &_system)
	{
		std::vector<std::vector<spin_ptr>> subspaces;		   // Collection of subspaces
		std::vector<spin_ptr> remainingSpins(_system.Spins()); // Get a list of all spins

		// Loop until all spins in the SpinSystem has been assigned to a subspace
		while (remainingSpins.size() > 0)
		{
			// Get the next unassigned spin
			spin_ptr nextSpin = remainingSpins.back();
			remainingSpins.pop_back();

			// Put the spin into a new subspace
			std::vector<spin_ptr> nextSubspace;
			nextSubspace.push_back(nextSpin);

			// Check whether any spins were added to the subspace in order to make it complete
			if (_system.CompleteSet(nextSubspace))
			{
				// Remove all the spins that were added to "nextSubspace" from the collection of unassigned spins
				for (auto i = nextSubspace.cbegin(); i != nextSubspace.cend(); i++)
					remainingSpins.erase(std::remove(remainingSpins.begin(), remainingSpins.end(), (*i)), remainingSpins.end());
			}

			// Add the subspace to the subspace collection
			subspaces.push_back(nextSubspace);
		}

		return subspaces;
	}

	// Checks whether the given subspace is complete with respect to the given SpinSystem (all interactions and transitions)
	bool IsCompleteSubspace(const std::vector<spin_ptr> &_subspace, const SpinSystem &_system)
	{
		return _system.IsComplete(_subspace);
	}

	// Checks whether the given subspace is complete with respect to the given interaction
	bool IsCompleteSubspace(const std::vector<spin_ptr> &_subspace, const interaction_ptr &_interaction)
	{
		if (_interaction == nullptr)
			return false;

		return _interaction->IsComplete(_subspace);
	}

	// Checks whether the given subspace is complete with respect to the given state (i.e. no outside entanglements)
	bool IsCompleteSubspace(const std::vector<spin_ptr> &_subspace, const state_ptr &_state)
	{
		if (_state == nullptr)
			return false;

		return _state->IsComplete(_subspace);
	}

	// Checks whether the given subspace is complete with respect to the given transition
	bool IsCompleteSubspace(const std::vector<spin_ptr> &_subspace, const transition_ptr &_transition)
	{
		if (_transition == nullptr)
			return false;

		return _transition->IsComplete(_subspace);
	}

	// Extends the subspace such that it is complete with respect to a SpinSystem. Returns true if the subspace was extend
	bool CompleteSubspace(std::vector<spin_ptr> &_subspace, const SpinSystem &_system)
	{
		return _system.CompleteSet(_subspace);
	}

	// Extends the subspace such that it is complete with respect to an Interactions. Returns true if the subspace was extend
	bool CompleteSubspace(std::vector<spin_ptr> &_subspace, const interaction_ptr &_interaction)
	{
		if (_interaction == nullptr)
			return false;

		return _interaction->CompleteSet(_subspace);
	}

	// Extends the subspace such that it is complete with respect to a State. Returns true if the subspace was extend
	bool CompleteSubspace(std::vector<spin_ptr> &_subspace, const state_ptr &_state)
	{
		if (_state == nullptr)
			return false;

		return _state->CompleteSet(_subspace);
	}

	// Extends the subspace such that it is complete with respect to a Transition. Returns true if the subspace was extend
	bool CompleteSubspace(std::vector<spin_ptr> &_subspace, const transition_ptr &_transition)
	{
		if (_transition == nullptr)
			return false;

		return _transition->CompleteSet(_subspace);
	}
	// -----------------------------------------------------
}
