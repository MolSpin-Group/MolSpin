/////////////////////////////////////////////////////////////////////////
// GeneralStateFrame (RunSection::General)
// ------------------
// Shared State-frame resolution for HS, SS and MultiSS General execution.
//
// A spatial orientation changes only objects declared in the molecular frame.
// Laboratory/fixed State projectors remain unchanged while the Hamiltonian is
// rotated.  The helpers below centralize the user-facing precedence rules so
// reaction sources and State observables cannot acquire backend-dependent
// orientation semantics.
//
// `eigen` is intentionally limited to a Thermal initial state.  A named State
// projector has no unique "eigen frame" without specifying which Hamiltonian,
// degeneracy convention, and state-to-eigenvector assignment define it.  The
// General backends therefore reject eigen for reactions/observables instead of
// silently treating it as a fixed laboratory projector.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_GeneralStateFrame
#define MOD_RunSection_General_GeneralStateFrame

#include "ObjectParser.h"
#include "SpinSystem.h"
#include "State.h"
#include "Transition.h"

#include <algorithm>
#include <cctype>
#include <string>

namespace RunSection::General
{
	inline SpinAPI::StateFrame ParseStateFrame(std::string value,
		SpinAPI::StateFrame fallback = SpinAPI::StateFrame::Fixed)
	{
		std::transform(value.begin(), value.end(), value.begin(),
			[](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		if (value == "molecular" || value == "mol" || value == "rotating")
			return SpinAPI::StateFrame::Molecular;
		if (value == "fixed" || value == "lab" || value == "laboratory" || value == "default")
			return SpinAPI::StateFrame::Fixed;
		if (value == "eigen" || value == "thermal")
			return SpinAPI::StateFrame::Eigen;
		return fallback;
	}

	inline SpinAPI::StateFrame SystemTransitionStateFrame(const SpinAPI::system_ptr &system)
	{
		if (system == nullptr || system->GetProperties() == nullptr)
			return SpinAPI::StateFrame::Fixed;
		std::string value;
		if (!(system->GetProperties()->Get("transitionstateframe", value) ||
			system->GetProperties()->Get("transition_state_frame", value)))
			return SpinAPI::StateFrame::Fixed;
		return ParseStateFrame(value, SpinAPI::StateFrame::Fixed);
	}

	inline SpinAPI::StateFrame TransitionStateFrame(const SpinAPI::system_ptr &system,
		const SpinAPI::transition_ptr &transition, bool target)
	{
		const SpinAPI::StateFrame fallback = SystemTransitionStateFrame(system);
		if (transition == nullptr || transition->Properties() == nullptr)
			return fallback;

		std::string value;
		const bool found = target
			? (transition->Properties()->Get("targetframe", value) ||
				transition->Properties()->Get("target_state_frame", value))
			: (transition->Properties()->Get("sourceframe", value) ||
				transition->Properties()->Get("source_state_frame", value));
		if (!found && !transition->Properties()->Get("transitionstateframe", value) &&
			!transition->Properties()->Get("stateframe", value))
			return fallback;
		return ParseStateFrame(value, fallback);
	}

	inline SpinAPI::StateFrame TransitionSourceStateFrame(const SpinAPI::system_ptr &system,
		const SpinAPI::transition_ptr &transition)
	{
		return TransitionStateFrame(system, transition, false);
	}

	inline SpinAPI::StateFrame TransitionTargetStateFrame(const SpinAPI::system_ptr &system,
		const SpinAPI::transition_ptr &transition)
	{
		return TransitionStateFrame(system, transition, true);
	}

	inline SpinAPI::StateFrame ObservableStateFrame(const SpinAPI::system_ptr &system,
		const SpinAPI::state_ptr &state)
	{
		SpinAPI::StateFrame fallback = SpinAPI::StateFrame::Fixed;
		std::string value;
		if (system != nullptr && system->GetProperties() != nullptr &&
			(system->GetProperties()->Get("observablestateframe", value) ||
				system->GetProperties()->Get("observable_state_frame", value)))
			fallback = ParseStateFrame(value, fallback);

		if (state == nullptr || state->Properties() == nullptr)
			return fallback;
		if (!(state->Properties()->Get("observableframe", value) ||
			state->Properties()->Get("observable_state_frame", value)))
			return fallback;
		return ParseStateFrame(value, fallback);
	}

	inline bool ValidateProjectorStateFrame(SpinAPI::StateFrame frame,
		const std::string &object, std::string &error)
	{
		if (frame != SpinAPI::StateFrame::Eigen)
			return true;
		error = object +
			" uses frame=eigen, which is defined only for a Thermal initial state; "
			"use frame=fixed or frame=molecular for named State projectors";
		return false;
	}
}

#endif
