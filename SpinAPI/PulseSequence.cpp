#include "PulseSequence.h"
#include "PulseSequence.h"
#include "PulseSequence.h"
/////////////////////////////////////////////////////////////////////////
// PulseSequence class (SpinAPI Module)
// ------------------
// PulseSequence for a spin system to define how pulses should be 
// 
// Molecular Spin Dynamics Software - developed by Luca Gerhards.
// (c) 2024 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "PulseSequence.h"
#include "Utility.h"
#include "ObjectParser.h"
#include <iostream>
#include "Pulse.h"
namespace SpinAPI
{
    PulseSequence::PulseSequence(std::string name, std::string contents)
        :properties(std::make_shared<MSDParser::ObjectParser>(name,contents))
    {  
        _validSequence = false;
        _sequenceParsedFlag = false;
        if(!this->properties->Get("offset", offset))
        {
            offset = 0.0;
            std::cerr << "No sequence offset defined, setting to 0.0" << std::endl;
        }
    }

    PulseSequence::~PulseSequence()
    {
        this->clear();
    }

    PulseSequence::PulseSequence(const PulseSequence& pulseseq)
        :properties(pulseseq.properties), sequence(pulseseq.sequence), tau_list(pulseseq.tau_list), _validSequence(pulseseq._validSequence), _sequenceParsedFlag(pulseseq._sequenceParsedFlag)
    {
    }

    bool PulseSequence::ParsePulseSequence(std::vector<pulse_ptr>& pulses)
    {
        std::string seq = "";
        _validSequence = false;
        if(!this->properties->Get("sequence", seq))
        {
            return false;
        }
        _sequenceParsedFlag = true;
        unsigned int errors = 0;
        std::unordered_map<std::string,double> cache;
        this->sequence.clear();
        auto trim = [](std::string &s) 
        {
            s.erase(0, s.find_first_not_of(" \t\r\n"));
            s.erase(s.find_last_not_of(" \t\r\n") + 1);
        };

        std::unordered_map<std::string, SpinAPI::pulse_ptr> pulse_map;
        pulse_map.reserve(pulses.size());
        for(const auto& p : pulses)
        {
            if (p) 
            {
                pulse_map[p->Name()] = p;
            }
        }

        std::stringstream ss(seq);
        std::string pulse_name;
        std::string tau_key;
        while (std::getline(ss, pulse_name, ',') && std::getline(ss, tau_key, ','))
        {

            trim(pulse_name);
            trim(tau_key);

            if (pulse_name.empty() || tau_key.empty()) 
            {
                continue;
            }

            double tau_value = 0.0;
        
            auto it = cache.find(tau_key);
            if (it != cache.end())
            {
                tau_value = it->second;
            }
            else
            {
                if (this->properties->Get(tau_key, tau_value))
                {
                    cache[tau_key] = tau_value;
                    this->tau_list[tau_key] = tau_value;
                }
                else
                {
                    errors += 1;
                }
            }

            auto it2 = pulse_map.find(pulse_name);
            if(it2 == pulse_map.end())
            {
                std::cerr << "Pulse: " << pulse_name << " not found";
                errors += 1;
                continue;
            }
            if(!it2->second->IsValid())
            {
                std::cerr << "Pulse: " << pulse_name << " is not valid";
                errors += 1;
                continue;
            }

            this->AddStep(it2->second,tau_key);
        }
        if (errors != 0)
            return false;
        _validSequence = true;
        return true;
    }

    std::pair<SpinAPI::pulse_ptr, double> PulseSequence::GetActivePulseAtTime(double abs_time) const
    {
        double track_time = this->offset;
        for (const auto& step : this->sequence)
        {
            auto[pulse, tau_key] = step;
            double pulse_duration = (pulse->Type() == SpinAPI::PulseType::InstantPulse) ? 0.0 : pulse->Pulsetime();

            if (abs_time >= track_time && abs_time < (track_time + pulse_duration))
            {
                double time_active = abs_time - track_time;
                return { pulse, time_active };
            }
            if(abs_time == track_time && pulse_duration == 0.0)
            {
                double time_active = abs_time - track_time;
                return {pulse, time_active};
            }

            track_time += pulse_duration;

            double gap_duration = 0.0;
            auto it = this->tau_list.find(tau_key);
            if (it != this->tau_list.end()) 
            {
                gap_duration = it->second;
            }

            if (abs_time >= track_time && abs_time < (track_time + gap_duration))
            {
                return { nullptr, 0.0 };
            }

            track_time += gap_duration;
        }

        return { nullptr, 0.0 };
    }

    const PulseSequence PulseSequence::operator=(const PulseSequence &seq)
    {   
        this->properties = std::make_shared<MSDParser::ObjectParser>(*(seq.properties));
        this->sequence = seq.sequence;
        this->tau_list = seq.tau_list;
        this->_validSequence = seq._validSequence;
        this->_sequenceParsedFlag = seq._sequenceParsedFlag;
        return (*this);
    }

    std::string PulseSequence::Name() const
    {
        return this->properties->Name();
    }
    
    bool PulseSequence::IsValid() const
    {
        return _validSequence;
    }
    
    std::shared_ptr<const MSDParser::ObjectParser> PulseSequence::Properties() const
    {
        return this->properties;
    }

    const std::unordered_map<std::string, double>& PulseSequence::Get_tau_list() const
    {
        return this->tau_list;
    }

    void PulseSequence::GetActionTargets(std::vector<RunSection::NamedActionScalar>&_scalers, std::vector<RunSection::NamedActionVector>&_vectors, const std::string &_system)
    {
        auto scalars = this->CreateActionScalars(_system);
        _scalers.insert(_scalers.end(),scalars.begin(), scalars.end());
    }

    std::vector<RunSection::NamedActionScalar> PulseSequence::CreateActionScalars(const std::string &_system)
    {
        std::vector<RunSection::NamedActionScalar> scalars;
        if(this->IsValid())
        {
            for (auto i = tau_list.begin(); i != tau_list.end(); i++)
            {
                RunSection::ActionScalar TauScaler = RunSection::ActionScalar(i->second, &CheckActionScalarTau);
                scalars.push_back(RunSection::NamedActionScalar(_system + "." + this->Name() + "." + i->first, TauScaler));
            }
        }
        return scalars;
    }

    bool CheckActionScalarTau(const double &_d)
	{
		return std::isfinite(_d);
	}
}