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
namespace SpinAPI
{
    PulseSequence::PulseSequence(std::string name, std::string contents)
        :properties(std::make_shared<MSDParser::ObjectParser(name,contents))
    {  

    }

    PulseSequence::~PulseSequence()
    {
        this->clear();
    }

    PulseSequence::PulseSequence(const PulseSequence& pulseseq)
        :properties(pulseseq.properties), :seqeunce(pulseseq.sequence)
    {
    }

    bool PulseSequence::ParsePulseSequence(std::vector<pulse_ptr>& pulses)
    {
        std::string seq = "";
        if(!this->properties->Get("sequence", seq))
        {
            return false;
        }

        unsigned int errors = 0;
        std::unordered_map<std::string,double> cache;
        this->sequence.clear();
        auto trim = [](std::string &s) 
        {
            s.erase(0, s.find_first_not_of(" \t\r\n"));
            s.erase(s.find_last_not_of(" \t\r\n") + 1);
        };

        std::unordered_map<std::string, SpinAPI::pulse_ptr> pulse_map;
        pulse_map.reserve(pulses.size())
        for(const auto& p : pulses)
        {
            if (p) 
            {
                pulse_map[p->Name()] = p;
            }
        }

        size_t pos =0;
        while((pos = seq.find('[',pos)) != std::string::npos)
        {
            size_t comma = seq.find(',', pos);
            size_t close_bracket = seq.find(']', comma);

            if (comma == std::string::npos || close_bracket == std::string::npos) 
            {
                break;
            }

            std::string pulse_name = seq.substr(pos + 1, comma - (pos + 1));
            std::string tau_key = seq.substr(comma + 1, close_bracket - (comma + 1));
            
            trim(pulse_name);
            trim(tau_key);

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
                    this-tau_list[tau_key] = tau_value;
                }
                else
                {
                    errors += 1;
                }
            }

            auto it = pulse_map.find(pulse_name);
            if(it == pulse_map.end())
            {
                std::cerr << "Pulse: " << pulse << " not found";
                errors += 1;
                continue;
            }

            this->AddStep(it->second,tau_key);
            pos = close_bracket + 1;
        }
        if errors != 0:
            return false;
        return true;
    }

    const PulseSequence PulseSequence::operator=(const PulseSequence &seq)
    {   
        this->properties = std::make_shared<MSDParser::ObjectParser>(*(seq.properties));
        this->sequence = seq.sequence;
        return (*this);
    }

    std::string SpinAPI::PulseSequence::Name() const
    {
        return this->properties->Name();
    }
    
    bool SpinAPI::PulseSequence::IsValid() const
    {
        return true;
    }
    
    std::shared_ptr<const MSDParser::ObjectParser> SpinAPI::PulseSequence::Properties() const
    {
        return this->properties;
    }
}
