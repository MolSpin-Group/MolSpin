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
#include <armadillo>

#include "SpinAPIfwd.h"
#include "SpinSpace.h"
#include "ActionTarget.h"
#include "SpinAPIDefines.h"
#include "MSDParserfwd.h"

namespace SpinAPI
{

    class SequenceObject
    {
        SpinAPI::pulse_ptr m_pulse_block = nullptr;
        SpinAPI::interaction_ptr m_interaction_block = nullptr;
        SpinAPI::transition_ptr m_transition_block = nullptr;
    public:
        void set_block(SpinAPI::pulse_ptr& block)
        {
            m_pulse_block = block;
        }
        void set_block(SpinAPI::interaction_ptr& block)
        {
            m_interaction_block = block;
        }
        void set_block(SpinAPI::transition_ptr& block)
        {
            m_transition_block = block;
        }

        std::tuple<SpinAPI::pulse_ptr, SpinAPI::interaction_ptr, SpinAPI::transition_ptr> get()
        {
            return std::tuple<SpinAPI::pulse_ptr, SpinAPI::interaction_ptr, SpinAPI::transition_ptr>{m_pulse_block,m_interaction_block,m_transition_block};
        }

        bool IsNullptr()
        {
            return (m_pulse_block == nullptr) && (m_transition_block == nullptr) && (m_interaction_block == nullptr);
        }
    };
	class PulseSequence
	{
    public:
        //using SequenceStep = std::tuple<SpinAPI::pulse_ptr, std::string>;
        using SequenceStep = std::tuple<SequenceObject, std::string>;
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

        bool ParsePulseSequence(std::vector<pulse_ptr>&, std::vector<interaction_ptr>&, std::vector<transition_ptr>&);
        std::pair<SequenceObject, double> GetActivePulseAtTime(double abs_time) const;

    private:
        std::vector<RunSection::NamedActionScalar> CreateActionScalars(const std::string&);
        void AddStep(SpinAPI::pulse_ptr ptr, std::string tau)
        {
            SequenceObject block;
            block.set_block(ptr);
            sequence.emplace_back(block, tau);
        }
        void AddStep(SpinAPI::interaction_ptr ptr, std::string tau)
        {
            SequenceObject block;
            block.set_block(ptr);
            sequence.emplace_back(block, tau);
        }
        void AddStep(SpinAPI::transition_ptr ptr, std::string tau)
        {
            SequenceObject block;
            block.set_block(ptr);
            sequence.emplace_back(block, tau);
        }
    };

    using PulseSequence_ptr = std::shared_ptr<PulseSequence>;

    bool CheckActionScalarTau(const double &);

    std::vector<arma::sp_cx_mat> GetPulseOperator(std::vector<std::pair<PulseSequence_ptr, std::shared_ptr<SpinSpace>>>, arma::cx_vec&, double);

}
#endif