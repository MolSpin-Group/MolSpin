/////////////////////////////////////////////////////////////////////////
// SubSystem implementation (SpinAPI Module)
// ------------------
// TODO: Write a description of the SubSystem class
//
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "SubSystem.h"
#include "SpinSystem.h"
#include "ObjectParser.h"
#include "Transition.h"
#include "Spin.h"

#include <iostream>

namespace SpinAPI
{
    SubSystem::SubSystem(std::string _name, std::string _contents, system_ptr _system)
        : m_name(_name), m_system(_system), m_spins({}), m_interactions({}), m_transtions({}), m_operators({}), m_pulses({}), m_properties(std::make_shared<MSDParser::ObjectParser>(_name, _contents))
    {
    }

    // SubSystem::SubSystem(SubSystem& _system)
    //{
    //     m_name = _system.m_name;
    //     m_properties = _system.m_properties;
    //     m_system = _system.m_system;
    //     m_spins = _system.m_spins;
    //     m_transtions = _system.m_transtions;
    //     m_interactions = _system.m_interactions;
    //     m_operators = _system.m_operators;
    //     m_pulses = _system.m_pulses;
    // }

    SubSystem::~SubSystem()
    {
    }

    std::string SubSystem::Name() const
    {
        return this->m_properties->Name();
    }

    bool SubSystem::Validate()
    {
        // checking that all the subsystem spins have been loaded
        std::vector<std::string> SubSystemSpins;
        {
            std::string SpinObjects = "";
            if (!this->m_properties->Get("spins", SpinObjects))
            {
                std::cout << "Error: No spin object list specified. Syntax is 'spins = ' " << std::endl;
                // std::cout << "Failed to load SubSystem " << this->Name() << std::endl;
                return false;
            }

            std::stringstream ss(SpinObjects);
            while (ss.good())
            {
                std::string Name;
                std::getline(ss, Name, ',');
                SubSystemSpins.push_back(Name);
            }
            {
                bool loaded = true;
                for (std::string i : SubSystemSpins)
                {
                    if (i == "")
                    {
                        continue;
                    }
                    if (m_system->spins_find(i) == nullptr)
                    {
                        std::cout << "Error: " << i << " is not a loaded spin object" << std::endl;
                        loaded &= false;
                        continue;
                    }
                    loaded &= true;
                }
                if (!loaded)
                {
                    // std::cout << "Failed to load SubSystem " << this->Name() << std::endl;
                    return false;
                }
            }
        }

        // checking that all transitions have been loaded
        std::vector<std::string> SubSystemTransitionsOut;
        {
            std::string TransitionsOut;
            if (!this->m_properties->Get("transitions_out", TransitionsOut))
            {
                std::cout << "INFO: No transitions leading out of subsystem " << this->Name() << " specified" << std::endl;
            }

            std::stringstream ss(TransitionsOut);
            while (ss.good())
            {
                std::string Name;
                std::getline(ss, Name, ',');
                SubSystemTransitionsOut.push_back(Name);
            }

            std::vector<int> RemovedTransitions;
            int index = 0;
            for (std::string i : SubSystemTransitionsOut)
            {
                if (i == "")
                {
                    RemovedTransitions.push_back(index);
                    continue;
                }
                if (m_system->transitions_find(i) == nullptr)
                {
                    std::cout << "Error: " << i << " is not a loaded transition object" << std::endl;
                    RemovedTransitions.push_back(index);
                }
                index++;
            }
            int removed = 0;
            for (auto i : RemovedTransitions)
            {
                SubSystemTransitionsOut.erase(SubSystemTransitionsOut.begin() + i - removed);
                removed++;
            }
        }
        // Checking that all transitions leading into the system have been loaded
        std::vector<std::string> SubSystemTransitionsIn;
        {
            std::string TransitionIn;
            if (!this->m_properties->Get("transitions_in", TransitionIn))
            {
                std::cout << "INFO: No transitions leading into the subsystem " << this->Name() << " specified" << std::endl;
            }

            std::stringstream ss(TransitionIn);
            while (ss.good())
            {
                std::string Name;
                std::getline(ss, Name, ',');
                SubSystemTransitionsIn.push_back(Name);
            }

            std::vector<int> RemovedTransitions;
            int index = 0;
            for (std::string i : SubSystemTransitionsIn)
            {
                if (i == "")
                {
                    RemovedTransitions.push_back(index);
                    continue;
                }
                if (m_system->transitions_find(i) == nullptr)
                {
                    std::cout << "Error: " << i << " is not a loaded transition object" << std::endl;
                    RemovedTransitions.push_back(index);
                }
                index++;
            }
            int removed = 0;
            for (auto i : RemovedTransitions)
            {
                SubSystemTransitionsIn.erase(SubSystemTransitionsIn.begin() + i - removed);
                removed++;
            }
        }
        // Checking that all interactions have been loaded
        std::vector<std::string> SubSystemInteractions;
        {
            std::string SystemInteractions;
            if (!this->m_properties->Get("interactions", SystemInteractions))
            {
                std::cout << "INFO: No interactions objects for subsystem " << this->Name() << " specified" << std::endl;
            }

            std::stringstream ss(SystemInteractions);
            while (ss.good())
            {
                std::string Name;
                std::getline(ss, Name, ',');
                SubSystemInteractions.push_back(Name);
            }

            std::vector<int> RemovedInteractions;
            int index = 0;
            for (std::string i : SubSystemInteractions)
            {
                if (i == "")
                {
                    RemovedInteractions.push_back(index);
                    continue;
                }
                if (m_system->interactions_find(i) == nullptr)
                {
                    std::cout << "Error: " << i << " is not a loaded interaction object" << std::endl;
                    RemovedInteractions.push_back(index);
                    continue;
                }

                // TODO: check that the spin objects are contained within the subsystem

                index++;
            }
            int removed = 0;
            for (auto i : RemovedInteractions)
            {
                SubSystemInteractions.erase(SubSystemInteractions.begin() + i - removed);
                removed++;
            }
        }
        // Checking that all operations have been loaded
        std::vector<std::string> SubSystemOperations;
        {
            std::string SystemOperations;
            if (!this->m_properties->Get("operators", SystemOperations))
            {
                std::cout << "INFO: No operators objects for subsystem " << this->Name() << " specified" << std::endl;
            }

            std::stringstream ss(SystemOperations);
            while (ss.good())
            {
                std::string Name;
                std::getline(ss, Name, ',');
                SubSystemOperations.push_back(Name);
            }

            std::vector<int> RemovedOperations;
            int index = 0;
            for (std::string i : SubSystemOperations)
            {
                if (i == "")
                {
                    RemovedOperations.push_back(index);
                    continue;
                }
                if (m_system->operators_find(i) == nullptr)
                {
                    std::cout << "Error: " << i << " is not a loaded operators object" << std::endl;
                    RemovedOperations.push_back(index);
                }
                index++;
            }
            int removed = 0;
            for (auto i : RemovedOperations)
            {
                SubSystemOperations.erase(SubSystemOperations.begin() + i - removed);
                removed++;
            }
        }
        // Checking that all pulses have been loaded
        std::vector<std::string> SubSystemPulses;
        {
            std::string SystemPulses;
            if (!this->m_properties->Get("pulses", SystemPulses))
            {
                std::cout << "INFO: No pulse objects for subsystem " << this->Name() << " specified" << std::endl;
            }

            std::stringstream ss(SystemPulses);
            while (ss.good())
            {
                std::string Name;
                std::getline(ss, Name, ',');
                SubSystemPulses.push_back(Name);
            }

            std::vector<int> RemovedPulses;
            int index = 0;
            for (std::string i : SubSystemPulses)
            {
                if (i == "")
                {
                    RemovedPulses.push_back(index);
                    continue;
                }
                if (m_system->pulses_find(i) == nullptr)
                {
                    std::cout << "Error: " << i << " is not a loaded pulse object" << std::endl;
                    RemovedPulses.push_back(index);
                }
                index++;
            }
            int removed = 0;
            for (auto i : RemovedPulses)
            {
                SubSystemPulses.erase(SubSystemPulses.begin() + i - removed);
                removed++;
            }
        }

        for (unsigned int i = 0; i < SubSystemSpins.size(); i++)
        {
            m_spins.push_back(std::make_shared<spin_ptr>(m_system->spins_find(SubSystemSpins[i])));
        }

        for (unsigned int i = 0; i < SubSystemTransitionsOut.size(); i++)
        {
            std::shared_ptr<transition_ptr> t = std::make_shared<transition_ptr>(m_system->transitions_find(SubSystemTransitionsOut[i]));
            Transition tr(t, t->get()->Name());
            tr.source = this->Name();
            m_transtions.push_back(tr);
        }
        for (unsigned int i = 0; i < SubSystemTransitionsIn.size(); i++)
        {
            std::shared_ptr<transition_ptr> t = std::make_shared<transition_ptr>(m_system->transitions_find(SubSystemTransitionsIn[i]));
            Transition tr(t, t->get()->Name());
            tr.target = this->Name();
            m_transtions.push_back(tr);
        }
        for (unsigned int i = 0; i < SubSystemInteractions.size(); i++)
        {
            m_interactions.push_back(std::make_shared<interaction_ptr>(m_system->interactions_find(SubSystemInteractions[i])));
        }

        // TODO: Pulses and interactions;

        return true;
    }

    std::vector<interaction_ptr> SubSystem::GetInteractions()
    {
        std::vector<interaction_ptr> _return;
        for (auto i : m_interactions)
        {
            _return.push_back(*i);
        }
        return _return;
    }

    std::vector<std::string> SubSystem::GetSpinNames()
    {
        std::vector<std::string> _return;
        for (auto i : m_spins)
        {
            _return.push_back(i->get()->Name());
        }
        return _return;
    }

    bool LinkTransitions(system_ptr system)
    {
        std::vector<SubSystem::Transition> AllTransitions;
        std::vector<int> data;
        int num = 0;
        for (auto ss : system->SubSystems())
        {
            for (auto tr : ss->GetTransitions())
            {
                AllTransitions.push_back(tr);
                num++;
            }
            data.push_back(num);
            num = 0;
        }

        int index1 = 0, index2 = 0;
        for (auto tr = AllTransitions.begin(); tr != AllTransitions.end(); tr++)
        {
            auto tr_ptr = tr->TransitionObject->get();
            for (auto tr2 = AllTransitions.begin(); tr2 != AllTransitions.end(); tr2++)
            {
                if (index1 == index2)
                {
                    index2++;
                    continue;
                }
                auto tr2_ptr = tr2->TransitionObject->get();
                std::string d = "";
                if (tr2->TransitionObject->get()->Properties()->Get("duplicate", d))
                    ;
                if (tr_ptr == tr2_ptr && d[0] != 't')
                {
                    tr->type = 1;
                    if (tr->source == "")
                    {
                        tr->source = tr2->source;
                        tr2->target = tr->target;
                    }
                    if (tr->target == "")
                    {
                        tr->target = tr2->target;
                        tr2->source = tr->source;
                    }

                    // put console message that transition was skipped

                    break;
                }
                index2++;
            }
            if ((unsigned int)index2 == AllTransitions.size() && tr->target == "")
            {
                tr->type = 0;
            }
            index2 = 0;
            index1++;
        }

        for (auto tr : AllTransitions)
        {
            if (tr.type == -1)
            {
                std::cout << "INFO: No source set for transition " << tr.name << " leading into subsystem " << tr.target << std::endl;
            }
        }

        int subsystem = 0;
        int TrNum = 0;
        for (auto dat : data)
        {
            for (int i = 0; i < dat; i++)
            {
                system->SubSystems()[subsystem]->ModifyTransitionVec(i, AllTransitions[TrNum]); // = AllTransitions[TrNum];
                TrNum++;
            }
            subsystem++;
        }
        return true;
    }
}
