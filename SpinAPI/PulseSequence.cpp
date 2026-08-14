/////////////////////////////////////////////////////////////////////////
// PulseSequence class (SpinAPI Module)
// ------------------
// PulseSequence for a spin system to define how pulses should be 
// 
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2024 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "PulseSequence.h"
#include "Utility.h"
#include "ObjectParser.h"
#include <iostream>
#include "Pulse.h"
#include "Interaction.h"
#include "Transition.h"
#include <type_traits>
namespace SpinAPI
{
    PulseSequence::PulseSequence(std::string name, std::string contents)
        : properties(std::make_shared<MSDParser::ObjectParser>(name, contents)),
          _sequenceParsedFlag(false), _validSequence(false), offset(0.0)
    {
        if(!this->properties->Get("offset", offset))
        {
            std::cerr << "No sequence offset defined, setting to 0.0" << std::endl;
        }
    }

    PulseSequence::~PulseSequence()
    {
        this->clear();
    }

    PulseSequence::PulseSequence(const PulseSequence& pulseseq)
        : properties(pulseseq.properties), sequence(pulseseq.sequence),
          tau_list(pulseseq.tau_list),
          _sequenceParsedFlag(pulseseq._sequenceParsedFlag),
          _validSequence(pulseseq._validSequence), offset(pulseseq.offset)
    {
    }

    bool PulseSequence::ParsePulseSequence(std::vector<pulse_ptr>& pulses, std::vector<interaction_ptr>& interactions, std::vector<transition_ptr>& transitions)
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
        std::unordered_map<std::string, SpinAPI::interaction_ptr> interaction_map;
        std::unordered_map<std::string, SpinAPI::transition_ptr> transition_map;
        pulse_map.reserve(pulses.size());
        interaction_map.reserve(interactions.size());
        transition_map.reserve(transitions.size());
        for(const auto& p : pulses)
        {
            if (p) 
            {
                pulse_map[p->Name()] = p;
            }
        }
        for(const auto& t : transitions)
        {
            if (t) 
            {
                transition_map[t->Name()] = t;
            }
        }
        for(const auto& i : interactions)
        {
            if (i) 
            {
                interaction_map[i->Name()] = i;
            }
        }
        std::stringstream ss(seq);
        std::string pulse_name;
        std::string tau_key;
        while (std::getline(ss, pulse_name, ',') && std::getline(ss, tau_key, ','))
        {
            tau_key = RunSection::lowercase(tau_key);

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


            auto process_step = [&](auto& target_map, const std::string& item, const std::string type, const auto& tau_key)
            {
                auto it2 = target_map.find(item);
                if(it2 == target_map.end()) 
                {
                    return false;
                }
                if(!it2->second->IsValid())
                {
                    std::cerr << type << ": " << item << " is not valid" << std::endl;
                    errors += 1;
                    return true;
                }
                
                using pt_type = typename std::decay_t<decltype(target_map)>::mapped_type::element_type;
                if constexpr(std::is_same_v<pt_type, SpinAPI::Pulse>)
                {
                    this->AddStep(it2->second, tau_key);
                    return true;
                }
                else
                {
                    if(type == "Interaction" || type == "Transition")
                    {
                        if(!it2->second->GetPulsed())
                        {
                            std::cerr << type << ": " << item << " is ill defined - no duration defined" << std::endl;
                            errors += 1;
                            return true;
                        }
                    }
                }

                this->AddStep(it2->second, tau_key);
                return true;
            };

            bool match = process_step(pulse_map, pulse_name, "Pulse", tau_key) ||
                         process_step(interaction_map, pulse_name, "Interaction", tau_key) ||
                         process_step(transition_map, pulse_name, "Transition", tau_key);
            if(!match)
            {
                std::cerr << "Sequence item: " << pulse_name << " not found" << std::endl;
                errors += 1;
            }
        }
        if (errors != 0)
            return false;
        _validSequence = true;
        return true;
    }

    std::pair<SequenceObject, double> PulseSequence::GetActivePulseAtTime(double abs_time) const
    {
        double track_time = this->offset;
        for (const auto& step : this->sequence)
        {
            auto[block, tau_key] = step;
            auto[pulse,interaction,transition] = block.get();
            double pulse_duration = 0.0;
            if (pulse != nullptr)
                pulse_duration = (pulse->Type() == SpinAPI::PulseType::InstantPulse) ? 0.0 : pulse->Pulsetime();
            else if(interaction != nullptr)
            {
                pulse_duration = interaction->ActiveTime();
                //interaction->SetActive(false);
            }
            else if(transition != nullptr)
            {
                pulse_duration = transition->ActiveTime();
                //transition->SetActive(false);
            }
            else
                return {SequenceObject(), 0.0};

            if (abs_time >= track_time && abs_time < (track_time + pulse_duration))
            {
                double time_active = abs_time - track_time;
                return { block, time_active };
            }
            if(abs_time == track_time && pulse_duration == 0.0)
            {
                double time_active = abs_time - track_time;
                return {block, time_active};
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
                if (transition)
                    transition->SetActive(false);
                else if(interaction)
                    interaction->SetActive(false);
                return {SequenceObject(), 0.0 };
            }

            track_time += gap_duration;
        }

        return {SequenceObject(), 0.0 };
    }

    const PulseSequence PulseSequence::operator=(const PulseSequence &seq)
    {   
        this->properties = std::make_shared<MSDParser::ObjectParser>(*(seq.properties));
        this->sequence = seq.sequence;
        this->tau_list = seq.tau_list;
        this->_validSequence = seq._validSequence;
        this->_sequenceParsedFlag = seq._sequenceParsedFlag;
        this->offset = seq.offset;
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
		return std::isfinite(_d) && _d >= 0.0;
	}

    std::vector<arma::sp_cx_mat> GetPulseOperator(std::vector<std::pair<PulseSequence_ptr, std::shared_ptr<SpinSpace>>> sequences, arma::cx_vec& rho, double CurrentTime) //rho is included for the case where we have a instant pulse 
    {
        //first get the active pulses
        std::vector<std::tuple<SequenceObject, double, std::shared_ptr<SpinSpace>>> active = {};
        std::vector<arma::sp_cx_mat> pulses = {};
        std::vector<unsigned int> active_seq = {};
        unsigned int idx = 0;
        for(const auto &seq : sequences)
        {
            pulses.push_back(arma::sp_cx_mat(seq.second->SpaceDimensions(),seq.second->SpaceDimensions()));
            auto p = seq.first->GetActivePulseAtTime(CurrentTime);
            if(p.first.IsNullptr())
            {
                idx += 1;
                continue;
            }
            auto temp = std::make_tuple(std::move(p.first), p.second, seq.second);
            active.push_back(temp);
            active_seq.push_back(idx);
            idx += 1;
        }

        for(auto i = active.begin(); i != active.end(); i++)
        {
            auto[seq_object, current_duration,space] = (*i);
            auto[pulse_event, interaction_event, transition_event] = seq_object.get();
            auto evaluate_pulse = [&](SpinAPI::pulse_ptr& pulse) {
                //check if it's a instant pulse
                auto pulse_type = pulse->Type();
                arma::sp_cx_mat pulse_op;
                if (pulse_type != SpinAPI::PulseType::MWPulse)
                {
                    if(!space->PulseOperator(pulse,pulse_op))
                    {
                        //error
                        return arma::sp_cx_mat(pulse_op.n_rows,pulse_op.n_cols);
                    }
                }
                if(pulse_type== SpinAPI::PulseType::InstantPulse)
                {
                    rho = pulse_op * rho;
                    return arma::sp_cx_mat(pulse_op.n_rows,pulse_op.n_cols);
                }
                else if(pulse_type == SpinAPI::PulseType::LongPulseStaticField)
                {
                    arma::sp_cx_mat A = arma::cx_double(0.0,-1.0) * pulse_op;
                    //std::cout << A << std::endl;
                    return A;
                }
                else if(pulse_type == SpinAPI::PulseType::LongPulse)
                {
                    double t = current_duration;
                    arma::sp_cx_mat A = arma::cx_double(0.0, -1.0) * pulse_op * std::cos(pulse->Frequency() * t);
                    //std::cout << A << std::endl;
                    return A;
                }
                else if(pulse_type == SpinAPI::PulseType::MWPulse)
                {
                    double t = current_duration;
                    if(!space->PulseOperator_mw(pulse,pulse_op,t))
                    {
                        return arma::sp_cx_mat(pulse_op.n_rows,pulse_op.n_cols);
                    }
                    arma::sp_cx_mat A = arma::cx_double(0.0, -1.0) * pulse_op;
                    return A;
                }
                else
                {
                    return arma::sp_cx_mat(pulse_op.n_rows,pulse_op.n_cols);
                }
            };

            auto evaluate_interaction = [&](SpinAPI::interaction_ptr& interaction) {
                interaction->SetActive(true);
            };
            auto evaluate_transition = [&](SpinAPI::transition_ptr& transition) {
                transition->SetActive(true);
            };
            
            idx = i - active.begin();
            if(pulse_event)
                pulses[active_seq[idx]] = evaluate_pulse(pulse_event);
            if(interaction_event)
                evaluate_interaction(interaction_event);
            else if(transition_event)
                evaluate_transition(transition_event);
        }
        return pulses;
    }
}
