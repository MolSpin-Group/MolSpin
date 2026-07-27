// File: SpinAPI/SpinSpace.cpp
/////////////////////////////////////////////////////////////////////////
// SpinSpace class (SpinAPI Module)
// ------------------
// Helper class that defines operators/matrices on the space spanned by
// a collection of spins. E.g. the Sz operator in the total Hilbert space.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "ObjectParser.h"
#include "State.h"
#include "Spin.h"
#include "Interaction.h"
#include "Transition.h"
#include "Operator.h"
#include "Pulse.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "Utility.h"

#include <array>
#include <cmath>

// Include additional source files
#include "SpinSpace/SpinSpace_management.cpp"
#include "SpinSpace/SpinSpace_states.cpp"
#include "SpinSpace/SpinSpace_operators.cpp"
#include "SpinSpace/SpinSpace_hamiltonians.cpp"
#include "SpinSpace/SpinSpace_transitions.cpp"
#include "SpinSpace/SpinSpace_relaxation.cpp"
#include "SpinSpace/SpinSpace_pulses.cpp"

namespace SpinAPI
{
	// -----------------------------------------------------
	// SpinSpace Constructors and Destructor
	// -----------------------------------------------------
	SpinSpace::SpinSpace() : useSuperspace(false), useFullTensorRotation(true), spins(), interactions(), transitions(), pulses(), time(0.0), trajectoryStep(0), useTrajectoryStep(false),
							 reactionOperators(ReactionOperatorType::Haberkorn)
	{
	}

	SpinSpace::SpinSpace(const spin_ptr &_spin) : useSuperspace(false), useFullTensorRotation(true), spins(), interactions(), transitions(), pulses(), time(0.0), trajectoryStep(0),
												  useTrajectoryStep(false), reactionOperators(ReactionOperatorType::Haberkorn)
	{
		this->spins.push_back(_spin);
	}

	SpinSpace::SpinSpace(const std::vector<spin_ptr> &_spinlist) : useSuperspace(false), useFullTensorRotation(true), spins(_spinlist), interactions(), transitions(), pulses(), time(0.0),
																   trajectoryStep(0), useTrajectoryStep(false), reactionOperators(ReactionOperatorType::Haberkorn)
	{
	}

	SpinSpace::SpinSpace(const std::shared_ptr<SpinSystem> &_system) : useSuperspace(false), useFullTensorRotation(true), spins(), interactions(), transitions(), pulses(), time(0.0),
																	   trajectoryStep(0), useTrajectoryStep(false), reactionOperators(ReactionOperatorType::Haberkorn)
	{
		this->Add(_system);
	}

	SpinSpace::SpinSpace(const SpinSystem &_system) : useSuperspace(false), useFullTensorRotation(true), spins(_system.Spins()), interactions(_system.Interactions()), transitions(_system.Transitions()), pulses(_system.Pulses()),
													  time(0.0), trajectoryStep(0), useTrajectoryStep(false), reactionOperators(ReactionOperatorType::Haberkorn) // TODO: Use move rather than copy for _system.Spins
	{
	}

	SpinSpace::SpinSpace(const SpinSpace &_space) : useSuperspace(_space.useSuperspace), useFullTensorRotation(_space.useFullTensorRotation), spins(_space.spins), interactions(_space.interactions), transitions(_space.transitions), pulses(_space.pulses),
													time(_space.time), trajectoryStep(_space.trajectoryStep), useTrajectoryStep(_space.useTrajectoryStep),
													reactionOperators(ReactionOperatorType::Haberkorn)
	{
	}

	SpinSpace::~SpinSpace()
	{
	}
	// -----------------------------------------------------
	// Operators
	// -----------------------------------------------------
	// Copy-assignment
	const SpinSpace &SpinSpace::operator=(const SpinSpace &_space)
	{
		this->useSuperspace = _space.useSuperspace;
		this->useFullTensorRotation = _space.useFullTensorRotation;
		this->spins = _space.spins;
		this->interactions = _space.interactions;
		this->transitions = _space.transitions;
		this->pulses = _space.pulses;
		this->time = _space.time;
		this->trajectoryStep = _space.trajectoryStep;
		this->useTrajectoryStep = _space.useTrajectoryStep;
		this->reactionOperators = _space.reactionOperators;

		return (*this);
	}
	// -----------------------------------------------------
	// Other public methods
	// -----------------------------------------------------
	// Returns the total dimensions of the current spin space (either Hilbert or superspace)
	unsigned int SpinSpace::SpaceDimensions() const
	{
		if (this->useSuperspace)
			return this->SuperSpaceDimensions();

		return this->HilbertSpaceDimensions();
	}

	// Returns the total dimensions of the Hilbert space
	unsigned int SpinSpace::HilbertSpaceDimensions() const
	{
		unsigned int dimensions = 1;
		for (auto i = this->spins.cbegin(); i != this->spins.cend(); i++)
			if ((*i)->Multiplicity() > 0)
				dimensions *= static_cast<unsigned int>((*i)->Multiplicity());

		return dimensions;
	}

	// Returns the total dimensions of the super space (the square of the corresponding Hilbert space)
	unsigned int SpinSpace::SuperSpaceDimensions() const
	{
		auto i = this->HilbertSpaceDimensions();
		return i * i;
	}

	// Checks whether any interactions has a time-dependence
	bool SpinSpace::HasTimedependentInteractions() const
	{
		for (auto j = this->interactions.cbegin(); j != this->interactions.cend(); j++)
		{
			if ((*j)->HasTimeDependence())
				return true;
		}

		return false;
	}

	// Checks whether any transitions has a time-dependence
	bool SpinSpace::HasTimedependentTransitions() const
	{
		for (auto j = this->transitions.cbegin(); j != this->transitions.cend(); j++)
		{
			if ((*j)->TrajectoryLength() > 0)
				return true;
		}

		return false;
	}

	// Set whether or not any matrices should be returned in superspace format, or the native Hilbert space
	bool SpinSpace::UseSuperoperatorSpace(bool _useSuperspace)
	{
		return (this->useSuperspace = _useSuperspace);
	}

	// Set whether rotated interactions keep off-diagonal tensor elements
	bool SpinSpace::UseFullTensorRotation(bool _useFullTensorRotation)
	{
		return (this->useFullTensorRotation = _useFullTensorRotation);
	}

	// Sets the reaction operator type
	// NOTE: Do NOT allow ReactionOperatorType::Unspecified to be set here
	bool SpinSpace::SetReactionOperatorType(const ReactionOperatorType &_operatorType)
	{
		if (_operatorType != ReactionOperatorType::Haberkorn && _operatorType != ReactionOperatorType::Lindblad)
			return false; // Other reaction operator types are not yet implemented!

		this->reactionOperators = _operatorType;
		return true;
	}

	// Sets the time - used by trajectories or time-dependent interactions
	bool SpinSpace::SetTime(double _time)
	{
		// TODO: Is useTrajectoryStep still needed?
		this->useTrajectoryStep = false;

		// Set the time of Interaction objects
		for (auto i = this->interactions.begin(); i != this->interactions.end(); i++)
			(*i)->SetTime(_time);

		// Set the time of Transition objects
		for (auto i = this->transitions.begin(); i != this->transitions.end(); i++)
			(*i)->SetTime(_time);

		// Set the time of Spin objects
		for (auto i = this->spins.begin(); i != this->spins.end(); i++)
			(*i)->SetTime(_time);

		// Store the new time
		// TODO: Do we need to store the time? So far only needed in SpinSpace::InteractionOperator, but this dependency should be removed in the future
		this->time = _time;

		return (!this->useTrajectoryStep);
	}

	// Sets the trajectory step - does not affect time dependent interactions
	bool SpinSpace::SetTrajectoryStep(unsigned int _step)
	{
		this->trajectoryStep = _step;
		this->useTrajectoryStep = true;

		// Set the trajectory step of Interaction objects
		for (auto i = this->interactions.begin(); i != this->interactions.end(); i++)
			(*i)->SetTrajectoryStep(_step);

		// Set the trajectory step of Transition objects
		for (auto i = this->transitions.begin(); i != this->transitions.end(); i++)
			(*i)->SetTrajectoryStep(_step);

		// Set the trajectory step of Spin objects
		for (auto i = this->spins.begin(); i != this->spins.end(); i++)
			(*i)->SetTrajectoryStep(_step);

		return this->useTrajectoryStep;
	}
	// -----------------------------------------------------
}
