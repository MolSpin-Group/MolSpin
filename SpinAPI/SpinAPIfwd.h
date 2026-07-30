/////////////////////////////////////////////////////////////////////////
// ForwardDeclarations (SpinAPI Module)
// ------------------
// Forward declarations of classes etc. to decrease file inter-
// dependencies and speedup compilation time.
//
// NOTE: Make sure this file is included AFTER the other includes!
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_SpinAPI_ForwardDeclarations
#define MOD_SpinAPI_ForwardDeclarations

#include <armadillo>
#include <memory>

namespace SpinAPI
{
#ifndef MOD_SpinAPI_Spin
	class Spin;
	using spin_ptr = std::shared_ptr<Spin>;
#endif

#ifndef MOD_SpinAPI_Interaction
	class Interaction;
	using interaction_ptr = std::shared_ptr<Interaction>;
#endif

#ifndef MOD_SpinAPI_Transition
	class Transition;
	using transition_ptr = std::shared_ptr<Transition>;
#endif

#ifndef MOD_SpinAPI_Operator
	class Operator;
	using operator_ptr = std::shared_ptr<Operator>;
#endif

#ifndef MOD_SpinAPI_Pulse
	class Pulse;
	using pulse_ptr = std::shared_ptr<Pulse>;
#endif

#ifndef MOD_SpinAPI_PulseSequence
	class PulseSequence;
	using PulseSequence_ptr = std::shared_ptr<PulseSequence>;
#endif

#ifndef MOD_SpinAPI_State
	class State;
	using state_ptr = std::shared_ptr<State>;
	using StateSeries = std::vector<std::pair<int, arma::cx_double>>;
	using StatePair = std::pair<spin_ptr, StateSeries>;
	using CompleteState = std::vector<StatePair>;
#endif

#ifndef MOD_SpinAPI_Tensor
	class Tensor;
#endif

#ifndef MOD_SpinAPI_StandardOutput
	class StandardOutput;
	using output_ptr = std::shared_ptr<StandardOutput>;
#endif

#ifndef MOD_SpinAPI_SubSystem
	class SubSystem;
	using subsystem_ptr = std::shared_ptr<SubSystem>;
#endif

#ifndef MOD_SpinAPI_SpinSystem
	class SpinSystem;
	using system_ptr = std::shared_ptr<SpinSystem>;
#endif

}

#endif
