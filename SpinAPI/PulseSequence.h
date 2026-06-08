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
#include <unordered_map>

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
        std::unordered_map<std::string, double> tau_list;
        bool _sequenceParsedFlag;
        bool _validSequence;
        double offset;
    public:
        PulseSequence(std::string, std::string);
        PulseSequence(const PulseSequence& );
        ~PulseSequence();

        const PulseSequence operator=(const PulseSequence& );

        std::string Name() const;
        bool IsValid() const;

        std::shared_ptr<const MSDParser::ObjectParser> Properties() const;
        void GetActionTargets(std::vector<RunSection::NamedActionScalar>&, std::vector<RunSection::NamedActionVector>&, const std::string&);
        const std::unordered_map<std::string, double>& Get_tau_list() const;
        double Get_offset() const { return offset; }

        size_t size() const noexcept {return sequence.size();}
        bool empty() const noexcept {return sequence.empty();}
        void clear() noexcept {sequence.clear();}

        auto begin() noexcept { return sequence.begin();}
        auto end() noexcept { return sequence.end();}

        auto begin() const noexcept { return sequence.begin();}
        auto end() const noexcept { return sequence.end();}

        auto cbegin() const noexcept { return sequence.cbegin(); }
        auto cend() const noexcept { return sequence.cend(); }

        bool ParsePulseSequence(std::vector<pulse_ptr>&);
        std::pair<SpinAPI::pulse_ptr, double> GetActivePulseAtTime(double abs_time) const;

    private:
        std::vector<RunSection::NamedActionScalar> CreateActionScalars(const std::string&);
        void AddStep(SpinAPI::pulse_ptr ptr, std::string tau)
        {
            sequence.emplace_back(std::move(ptr), tau);
        }
    };

    using PulseSequence_ptr = std::shared_ptr<PulseSequence>;

    bool CheckActionScalarTau(const double &);

}
#endif