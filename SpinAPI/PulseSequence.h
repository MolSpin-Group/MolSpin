/////////////////////////////////////////////////////////////////////////
// PulseSequence class (SpinAPI Module)
// ------------------
// PulseSequence for a spin system to define how pulses should be 
// 
// Molecular Spin Dynamics Software - developed by Luca Gerhards.
// (c) 2024 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_SpinAPI_PulseSequence
#define MOD_SpinAPI_PulseSequence

#include <memory>
#include <vector>

#include "SpinAPIfwd.h"
#include "ActionTarget.h"
#include "SpinAPIDefines.h"
#include "MSDParserfwd.h"

namespace SpinAPI
{
	class PulseSequence
	{
    public:
        using SequenceStep = std::tuple<SpinAPI::pulse_ptr, std::string>;
    private:
        std::shared_ptr<MSDParser::ObjectParser> properties;
        std::vector<SequenceStep> sequence;
        std::unorderd_map<std::string, double> tau_list;
        bool valid = true;
    public:
        PulseSequence(std::string, std::string);
        PulseSequence(const PulseSequence& )
        ~PulseSequence();

        const PulseSequence operator=(const PulseSequence& );

        std::string Name() const;
        bool IsValid() const;

        std::shared_ptr<const MSDParser::ObjectParser> Properties() const;
        void GetActionTargets(std::vector<RunSection::NamedActionScalar>&, std::vector<RunSection::NamedActionVector>&, const std::string&);

        size_t size() const noexcept {return sequence.size();}
        bool empty() const noexcept {return sequence.empty();}
        void clear() noexcept {sequence.clear();}

        auto begin() noexcept {sequence.begin();}
        auto end() noexcept {sequence.end();}

        auto begin() const noexcept {sequence.begin();}
        auto end() const noexcept {sequence.end();}

        auto cbegin() const noexcept { return sequence.cbegin(); }
        auto cend() const noexcept { return sequence.cend(); }

    private:
        std::vector<RunSection::NamedActionScalar> CreateActionScalars(const std::string&);
        bool ParsePulseSequence(std::vector<pulse_ptr>&);
        void AddStep(SpinAPI::pulse_ptr ptr, std::string tau)
        {
            sequence.emplace_back(std::move(ptr), tau);
        }
    }

using PulseSequence_ptr = std::shared<PulseSequence>;

}
#endif