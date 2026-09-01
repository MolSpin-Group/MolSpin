/////////////////////////////////////////////////////////////////////////
// HybridNuclearResonancePartitionBuilder
// ------------------
// Builds the canonical physical hybrid partition from an EXPLICIT list of
// perturbative nuclei. It does not choose a perturbative partition from
// coupling strengths, microwave frequency, isotope, or any other heuristic.
//
// The complete SpinSystem interaction set is classified structurally so that
// HybridNuclearResonancePreparation retains its exact "every interaction once"
// ownership contract. Task-level Hamiltonian subsets must therefore be checked
// by the caller before using this builder.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_Resonance_HybridNuclearResonancePartitionBuilder
#define MOD_RunSection_General_Resonance_HybridNuclearResonancePartitionBuilder

#include "HybridNuclearResonancePreparation.h"
#include "Interaction.h"
#include "Spin.h"
#include "SpinSystem.h"
#include "State.h"

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace RunSection::General::Resonance
{
    struct HybridNuclearResonanceExplicitNucleus
    {
        SpinAPI::spin_ptr nucleus;
        double overlapThreshold = 1.0e-14;
        bool fieldIndependentProjection = false;
    };

    struct HybridNuclearResonanceExplicitPartitionRequest
    {
        SpinAPI::system_ptr system;

        // Explicit physical choice. The exact core is the complement of these
        // nuclei in system->Spins(); no strength-based choice is made here.
        std::vector<HybridNuclearResonanceExplicitNucleus>
            perturbativeNuclei;

        // Explicit dH/dB subset. Every entry must be one of the physical
        // SpinSystem interactions and is classified into exact-core or one
        // perturbative nuclear factor by the same ownership topology.
        std::vector<SpinAPI::interaction_ptr> fieldInteractions;

        // Detection-spin/Zeeman pairing remains caller-owned.
        std::vector<ResonanceMagneticMomentTerm> detectionTerms;

        SpinAPI::state_ptr exactCoreState;
        bool fullTensorRotation = true;

        double minimumCumulativeOverlapWeight = 0.0;
        std::size_t maximumComponentsPerCoreTransition = 0;
        double mergeFrequencyToleranceRadNs = 0.0;
    };

    class HybridNuclearResonancePartitionBuilder
    {
    private:
        template <typename T>
        static bool Contains(
            const std::vector<std::shared_ptr<T>> &values,
            const std::shared_ptr<T> &value)
        {
            return
                std::find(values.begin(),values.end(),value) !=
                values.end();
        }

        template <typename T>
        static bool Unique(
            const std::vector<std::shared_ptr<T>> &values)
        {
            for (std::size_t i=0;i<values.size();++i)
            {
                if (values[i]==nullptr)
                    return false;
                for (std::size_t j=i+1;j<values.size();++j)
                {
                    if (values[i]==values[j])
                        return false;
                }
            }
            return true;
        }

        static bool UniqueExplicitNuclei(
            const std::vector<
                HybridNuclearResonanceExplicitNucleus> &nuclei)
        {
            if (nuclei.empty())
                return false;

            for (std::size_t i=0;i<nuclei.size();++i)
            {
                if (nuclei[i].nucleus==nullptr ||
                    !std::isfinite(nuclei[i].overlapThreshold) ||
                    nuclei[i].overlapThreshold<0.0 ||
                    nuclei[i].overlapThreshold>1.0)
                    return false;

                for (std::size_t j=i+1;j<nuclei.size();++j)
                {
                    if (nuclei[i].nucleus==nuclei[j].nucleus)
                        return false;
                }
            }
            return true;
        }

        static int PerturbativeIndex(
            const std::vector<
                HybridNuclearResonanceExplicitNucleus> &nuclei,
            const SpinAPI::spin_ptr &spin)
        {
            for (std::size_t i=0;i<nuclei.size();++i)
            {
                if (nuclei[i].nucleus==spin)
                    return static_cast<int>(i);
            }
            return -1;
        }

        static bool AllSpinsInExactCore(
            const std::vector<SpinAPI::spin_ptr> &spins,
            const std::vector<SpinAPI::spin_ptr> &core)
        {
            for (const auto &spin:spins)
            {
                if (spin==nullptr || !Contains(core,spin))
                    return false;
            }
            return true;
        }

        static bool IsCoreOnlyInteraction(
            const SpinAPI::interaction_ptr &interaction,
            const std::vector<SpinAPI::spin_ptr> &core)
        {
            if (interaction==nullptr)
                return false;

            const auto g1=interaction->Group1();
            const auto g2=interaction->Group2();
            return
                !g1.empty() &&
                AllSpinsInExactCore(g1,core) &&
                AllSpinsInExactCore(g2,core);
        }

        static bool IsNucleusOnlyInteraction(
            const SpinAPI::interaction_ptr &interaction,
            const SpinAPI::spin_ptr &nucleus)
        {
            if (interaction==nullptr || nucleus==nullptr)
                return false;

            const auto g1=interaction->Group1();
            const auto g2=interaction->Group2();
            if (g1.empty() || !g2.empty())
                return false;

            for (const auto &spin:g1)
            {
                if (spin!=nucleus)
                    return false;
            }
            return true;
        }

        static bool IsCoreNucleusHyperfine(
            const SpinAPI::interaction_ptr &interaction,
            const SpinAPI::spin_ptr &nucleus,
            const std::vector<SpinAPI::spin_ptr> &core)
        {
            if (interaction==nullptr ||
                nucleus==nullptr ||
                interaction->Type()!=
                    SpinAPI::InteractionType::DoubleSpin)
                return false;

            const auto g1=interaction->Group1();
            const auto g2=interaction->Group2();

            const bool nucleusLeft=
                g1.size()==1 &&
                g1.front()==nucleus &&
                !g2.empty() &&
                AllSpinsInExactCore(g2,core);

            const bool nucleusRight=
                g2.size()==1 &&
                g2.front()==nucleus &&
                !g1.empty() &&
                AllSpinsInExactCore(g1,core);

            return nucleusLeft != nucleusRight;
        }

        static bool InteractionTouchesNucleus(
            const SpinAPI::interaction_ptr &interaction,
            const SpinAPI::spin_ptr &nucleus)
        {
            if (interaction==nullptr || nucleus==nullptr)
                return false;

            const auto g1=interaction->Group1();
            const auto g2=interaction->Group2();
            return
                std::find(g1.begin(),g1.end(),nucleus)!=g1.end() ||
                std::find(g2.begin(),g2.end(),nucleus)!=g2.end();
        }

        static bool StateExcludesNucleus(
            const SpinAPI::state_ptr &state,
            const SpinAPI::spin_ptr &nucleus)
        {
            if (state==nullptr || nucleus==nullptr)
                return false;

            std::vector<std::pair<int,arma::cx_double>> series;
            bool coupled=false;
            const bool present=
                state->GetStateSeries(nucleus,series,coupled);
            return !present && !coupled;
        }

    public:
        static bool Build(
            const HybridNuclearResonanceExplicitPartitionRequest &request,
            HybridNuclearResonancePartition &partition,
            std::string &error)
        {
            error.clear();
            partition=HybridNuclearResonancePartition{};

            if (request.system==nullptr)
            {
                error=
                    "explicit hybrid partition requires a SpinSystem";
                return false;
            }
            if (!UniqueExplicitNuclei(
                    request.perturbativeNuclei))
            {
                error=
                    "explicit hybrid perturbative nuclei must be non-empty, unique, and valid";
                return false;
            }
            if (!Unique(request.fieldInteractions))
            {
                error=
                    "explicit hybrid field interactions must be unique and non-null";
                return false;
            }
            if (request.exactCoreState==nullptr ||
                !request.system->Contains(request.exactCoreState))
            {
                error=
                    "explicit hybrid exact-core state is missing from the SpinSystem";
                return false;
            }
            if (!std::isfinite(
                    request.minimumCumulativeOverlapWeight) ||
                request.minimumCumulativeOverlapWeight<0.0 ||
                request.minimumCumulativeOverlapWeight>1.0 ||
                !std::isfinite(
                    request.mergeFrequencyToleranceRadNs) ||
                request.mergeFrequencyToleranceRadNs<0.0)
            {
                error=
                    "explicit hybrid composition controls are invalid";
                return false;
            }

            const auto systemSpins=request.system->Spins();
            if (systemSpins.empty())
            {
                error=
                    "explicit hybrid SpinSystem has no spins";
                return false;
            }

            for (const auto &spec:request.perturbativeNuclei)
            {
                if (!request.system->Contains(spec.nucleus) ||
                    spec.nucleus->Type()!=SpinAPI::SpinType::Nucleus)
                {
                    error=
                        "explicit hybrid perturbative spin must be a nuclear SpinSystem member";
                    return false;
                }
                if (!StateExcludesNucleus(
                        request.exactCoreState,spec.nucleus))
                {
                    error=
                        "explicit hybrid exact-core state must not specify a perturbative nucleus";
                    return false;
                }
            }

            partition.system=request.system;
            partition.exactCoreState=request.exactCoreState;
            partition.fullTensorRotation=
                request.fullTensorRotation;
            partition.minimumCumulativeOverlapWeight=
                request.minimumCumulativeOverlapWeight;
            partition.maximumComponentsPerCoreTransition=
                request.maximumComponentsPerCoreTransition;
            partition.mergeFrequencyToleranceRadNs=
                request.mergeFrequencyToleranceRadNs;

            for (const auto &spin:systemSpins)
            {
                if (PerturbativeIndex(
                        request.perturbativeNuclei,spin)<0)
                    partition.exactCoreSpins.push_back(spin);
            }
            if (partition.exactCoreSpins.empty())
            {
                error=
                    "explicit hybrid partition requires a non-empty exact core";
                return false;
            }

            partition.nuclei.resize(
                request.perturbativeNuclei.size());
            for (std::size_t i=0;
                 i<request.perturbativeNuclei.size();++i)
            {
                partition.nuclei[i].nucleus=
                    request.perturbativeNuclei[i].nucleus;
                partition.nuclei[i].overlapThreshold=
                    request.perturbativeNuclei[i].overlapThreshold;
                partition.nuclei[i].fieldIndependentProjection=
                    request.perturbativeNuclei[i].
                        fieldIndependentProjection;
            }

            const auto systemInteractions=
                request.system->Interactions();
            if (systemInteractions.empty())
            {
                error=
                    "explicit hybrid SpinSystem has no interactions";
                return false;
            }

            // Classify every physical SpinSystem interaction exactly once.
            for (const auto &interaction:systemInteractions)
            {
                if (interaction==nullptr ||
                    interaction->HasTimeDependence())
                {
                    error=
                        "explicit hybrid partition requires static non-null interactions";
                    return false;
                }

                if (IsCoreOnlyInteraction(
                        interaction,partition.exactCoreSpins))
                {
                    partition.exactCoreInteractions.
                        push_back(interaction);
                    continue;
                }

                int touchedFactor=-1;
                int touchedCount=0;
                for (std::size_t i=0;
                     i<partition.nuclei.size();++i)
                {
                    if (InteractionTouchesNucleus(
                            interaction,
                            partition.nuclei[i].nucleus))
                    {
                        touchedFactor=
                            static_cast<int>(i);
                        ++touchedCount;
                    }
                }

                if (touchedCount!=1)
                {
                    error=
                        "explicit hybrid interaction couples multiple perturbative factors or has unowned spins";
                    return false;
                }

                auto &factor=
                    partition.nuclei[
                        static_cast<std::size_t>(
                            touchedFactor)];

                if (IsNucleusOnlyInteraction(
                        interaction,factor.nucleus))
                {
                    factor.nuclearInteractions.
                        push_back(interaction);
                    continue;
                }

                if (IsCoreNucleusHyperfine(
                        interaction,factor.nucleus,
                        partition.exactCoreSpins))
                {
                    if (factor.hyperfine!=nullptr)
                    {
                        error=
                            "explicit hybrid partition requires exactly one core-nucleus hyperfine interaction per perturbative nucleus";
                        return false;
                    }
                    factor.hyperfine=interaction;
                    continue;
                }

                error=
                    "explicit hybrid interaction topology is not representable by independent perturbative nuclear factors";
                return false;
            }

            for (const auto &factor:partition.nuclei)
            {
                if (factor.hyperfine==nullptr)
                {
                    error=
                        "explicit hybrid partition requires exactly one core-nucleus hyperfine interaction per perturbative nucleus";
                    return false;
                }
            }

            // Field interactions are an explicit subset of already-classified
            // physical interactions. A core-nucleus HFC can never be dH/dB.
            for (const auto &field:request.fieldInteractions)
            {
                if (!request.system->Contains(field))
                {
                    error=
                        "explicit hybrid field interaction is missing from the SpinSystem";
                    return false;
                }

                if (Contains(
                        partition.exactCoreInteractions,field))
                {
                    partition.exactCoreFieldInteractions.
                        push_back(field);
                    continue;
                }

                bool assigned=false;
                for (auto &factor:partition.nuclei)
                {
                    if (Contains(
                            factor.nuclearInteractions,field))
                    {
                        factor.nuclearFieldInteractions.
                            push_back(field);
                        assigned=true;
                        break;
                    }
                }

                if (!assigned)
                {
                    error=
                        "explicit hybrid field interaction must belong wholly to the exact core or one perturbative nucleus";
                    return false;
                }
            }

            // Detection terms must remain exact-core observables and their
            // Zeeman interactions must be in the declared field subset.
            if (request.detectionTerms.empty())
            {
                error=
                    "explicit hybrid partition requires at least one detection term";
                return false;
            }
            for (const auto &term:request.detectionTerms)
            {
                if (term.spin==nullptr ||
                    term.zeeman==nullptr ||
                    !Contains(
                        partition.exactCoreSpins,
                        term.spin))
                {
                    error=
                        "explicit hybrid detection spin must belong to the exact core";
                    return false;
                }
                if (!Contains(
                        partition.exactCoreInteractions,
                        term.zeeman) ||
                    !Contains(
                        request.fieldInteractions,
                        term.zeeman))
                {
                    error=
                        "explicit hybrid detection Zeeman interaction must be an exact-core field interaction";
                    return false;
                }
            }
            partition.detectionTerms=request.detectionTerms;

            return true;
        }
    };
}

#endif
