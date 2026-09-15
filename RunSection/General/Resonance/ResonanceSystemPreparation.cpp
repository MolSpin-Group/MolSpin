/////////////////////////////////////////////////////////////////////////
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "ResonanceSystemPreparation.h"
#include "SpinSystem.h"
#include "Spin.h"
#include "Interaction.h"
#include "State.h"
#include "ObjectParser.h"
#include "GeneralResonanceHamiltonian.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace RunSection::General::Resonance
{
    void FieldSyncGuard::Apply(const std::vector<SpinAPI::interaction_ptr> &_interactions,
                               const arma::vec &_field)
    {
        saved.clear();
        saved.reserve(_interactions.size());
        for (const auto &inter : _interactions)
        {
            if (inter == nullptr)
                continue;
            const arma::vec current = inter->Field();
            if (current.n_elem != 3 || !current.is_finite())
                continue;
            saved.emplace_back(inter, current);
            arma::vec tmp = _field;
            inter->SetField(tmp);
        }
    }
    FieldSyncGuard::~FieldSyncGuard()
    {
        for (auto &entry : saved)
        {
            arma::vec tmp = entry.second;
            entry.first->SetField(tmp);
        }
    }
    namespace SystemPreparation
    {
        bool ResolveHamiltonian(const ResonanceExecutionPlan &plan, const SpinAPI::system_ptr &system,
                                std::vector<std::string> &names, std::string &error)
        {
            error.clear();
            names = plan.hamiltonianH0list;
            if (!system)
            {
                error = "null resonance spin system";
                return false;
            }
            if (names.empty())
                for (const auto &interaction : system->Interactions())
                    if (interaction && SpinAPI::IsStatic(*interaction))
                        names.push_back(interaction->Name());
            if (names.empty())
            {
                error = "resonance Hamiltonian has no static interactions";
                return false;
            }
            for (const auto &name : names)
            {
                auto interaction = system->interactions_find(name);
                if (!interaction || !SpinAPI::IsStatic(*interaction) ||
                    std::count(names.begin(), names.end(), name) != 1)
                {
                    error = "unknown, dynamic or duplicate resonance Hamiltonian interaction: " + name;
                    return false;
                }
            }
            return true;
        }

        bool BuildHamiltonian(SpinAPI::SpinSpace &space, const std::vector<std::string> &names,
                              const arma::mat &rotation, arma::sp_cx_mat &h, std::string &error)
        {
            // Use the same General HS Hamiltonian builder as propagation, with the
            // full Hamiltonian approximation required by this resonance task.
            HS::HSExecutionPlan hp;
            hp.hasH0List = true;
            hp.h0List = names;
            hp.approximation = SpinAPI::HamiltonianApproximation::Full;
            HS::HSOrientation orientation;
            orientation.frameToLab = rotation;
            GeneralResonanceHamiltonian builder(hp, space);
            return builder.Build(orientation, h, error);
        }

        bool PrepareState(const ResonanceExecutionPlan &plan, const SpinAPI::system_ptr &system,
                          SpinAPI::SpinSpace &space, HS::HSPreparedState &state, std::string &error)
        {
            error.clear();
            state = HS::HSPreparedState();
            if (!system)
            {
                error = "null resonance state system";
                return false;
            }
            state.frame = system->InitialStateFrame();
            if (state.frame == SpinAPI::StateFrame::Eigen)
            {
                const auto states = system->InitialState();
                if (!plan.initialStateName.empty() || states.size() != 1 || states.front() != nullptr)
                {
                    error = "frame=eigen requires one Thermal SpinSystem initial state";
                    return false;
                }
                state.orientationSpecificThermal = true;
                state.thermalHamiltonian = system->ThermalHamiltonianList();
                state.thermalTemperature = system->Temperature();
                if (!std::isfinite(state.thermalTemperature) || !(state.thermalTemperature > 0))
                {
                    error = "thermal resonance temperature must be finite and positive";
                    return false;
                }
                for (const auto &name : state.thermalHamiltonian)
                    if (!system->interactions_find(name))
                    {
                        error = "unknown thermal interaction: " + name;
                        return false;
                    }
                return true;
            }
            if (!plan.initialStateName.empty())
            {
                auto named = system->states_find(plan.initialStateName);
                if (!named || !space.GetState(named, state.density))
                {
                    error = "invalid task-level initial state: " + plan.initialStateName;
                    return false;
                }
                const auto tr = arma::trace(state.density);
                if (!std::isfinite(tr.real()) || std::abs(tr) == 0)
                {
                    error = "initial state has invalid trace";
                    return false;
                }
                state.density /= tr;
            }
            else
            {
                for (const auto &named : system->InitialState())
                    if (!named)
                    {
                        error = "Thermal resonance initial state requires frame=eigen";
                        return false;
                    }
                if (!HS::HSStatePreparation::BuildInitialDensity(system, space, state.density, error, false))
                    return false;
            }
            return true;
        }

        bool OrientState(SpinAPI::SpinSpace &space, const HS::HSPreparedState &state,
                         const arma::mat &rotation, arma::cx_mat &density, std::string &error)
        {
            HS::HSExecutionPlan hp;
            HS::HSOrientation orientation;
            orientation.frameToLab = rotation;
            return HS::HSStatePreparation::PrepareDensityForOrientation(hp, space, state, orientation,
                                                                        density, error);
        }

        bool PrepareMoments(const std::vector<SpinAPI::spin_ptr> &spins,
                            const std::vector<SpinAPI::interaction_ptr> &zeeman,
                            std::vector<ResonanceMagneticMomentTerm> &terms, std::string &error)
        {
            error.clear();
            terms.clear();
            for (const auto &spin : spins)
            {
                auto interaction = FindZeemanForSpin(spin, zeeman);
                if (!interaction)
                {
                    error = "no Zeeman interaction owns detection spin " + spin->Name();
                    return false;
                }
                ResonanceMagneticMomentTerm term;
                term.spin = spin;
                term.zeeman = interaction;
                terms.push_back(term);
            }
            return !terms.empty();
        }
        bool IsZeemanInteraction(const SpinAPI::interaction_ptr &inter)
        {
            if (inter == nullptr)
                return false;
            if (!SpinAPI::IsStatic(*inter))
                return false;
            if (inter->Type() != SpinAPI::InteractionType::SingleSpin)
                return false;
            const arma::vec field = inter->Field();
            return (field.n_elem == 3 && field.is_finite());
        }
        std::vector<SpinAPI::interaction_ptr> CollectZeemanInteractions(
            const SpinAPI::system_ptr &system, const std::vector<std::string> &h0list)
        {
            std::vector<SpinAPI::interaction_ptr> out;
            if (system == nullptr)
                return out;

            out.reserve(h0list.size());
            for (const auto &name : h0list)
            {
                auto inter = system->interactions_find(name);
                if (!IsZeemanInteraction(inter))
                    continue;
                out.push_back(inter);
            }

            std::sort(out.begin(), out.end(), [](const auto &a, const auto &b) { return a.get() < b.get(); });
            out.erase(std::unique(out.begin(), out.end()), out.end());
            return out;
        }
        SpinAPI::interaction_ptr FindZeemanForSpin(const SpinAPI::spin_ptr &spin,
                                                   const std::vector<SpinAPI::interaction_ptr> &zeemanList)
        {
            if (spin == nullptr)
                return nullptr;
            for (const auto &inter : zeemanList)
            {
                if (inter == nullptr)
                    continue;
                const auto group = inter->Group1();
                if (std::find(group.begin(), group.end(), spin) != group.end())
                    return inter;
            }
            return nullptr;
        }
        bool ResolveFieldInteraction(const ResonanceExecutionPlan &plan, const SpinAPI::system_ptr &_system,
                                     SpinAPI::interaction_ptr &_fieldInteraction)
        {
            _fieldInteraction = nullptr;
            if (_system == nullptr)
                return false;

            if (!plan.fieldInteractionName.empty())
            {
                _fieldInteraction = _system->interactions_find(plan.fieldInteractionName);
                return IsZeemanInteraction(_fieldInteraction);
            }

            if (_fieldInteraction == nullptr)
            {
                for (auto inter = _system->interactions_cbegin(); inter != _system->interactions_cend();
                     inter++)
                {
                    std::string type;
                    if ((*inter)->Properties()->Get("type", type))
                    {
                        type = ResonanceLowercase(type);
                        if (type == "zeeman")
                        {
                            _fieldInteraction = (*inter);
                            break;
                        }
                    }
                }
            }

            return IsZeemanInteraction(_fieldInteraction);
        }
        bool ResolveDetectionSpins(const ResonanceExecutionPlan &plan, const SpinAPI::system_ptr &_system,
                                   const SpinAPI::interaction_ptr &_fieldInteraction,
                                   std::vector<SpinAPI::spin_ptr> &_spins,
                                   std::vector<std::string> &_spinNames)
        {
            _spins.clear();
            _spinNames.clear();
            if (_system == nullptr)
                return false;

            auto add_spin = [&](const SpinAPI::spin_ptr &spin) -> bool
            {
                if (spin == nullptr)
                    return false;
                for (const auto &existing : _spins)
                {
                    if (existing == spin)
                        return true;
                }
                _spins.push_back(spin);
                return true;
            };

            // Detection spins define which magnetic dipole operators contribute to the
            // reported per-spin channels. If the user does not specify them explicitly,
            // we fall back to the Zeeman interaction and finally to all spins.
            if (!plan.detectSpinNames.empty())
            {
                for (const auto &name : plan.detectSpinNames)
                {
                    auto spin = _system->spins_find(name);
                    if (spin == nullptr)
                        return false;
                    add_spin(spin);
                }
            }
            else if (_fieldInteraction != nullptr)
            {
                auto group = _fieldInteraction->Group1();
                for (const auto &spin : group)
                {
                    add_spin(spin);
                }
            }

            if (_spins.empty())
            {
                auto allSpins = _system->Spins();
                for (const auto &spin : allSpins)
                    add_spin(spin);
            }

            if (_spins.empty())
                return false;

            for (const auto &spin : _spins)
                _spinNames.push_back(spin->Name());

            return true;
        }
    } // namespace SystemPreparation
} // namespace RunSection::General::Resonance
