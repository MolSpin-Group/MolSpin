/////////////////////////////////////////////////////////////////////////
// HybridNuclearResonancePreparation (RunSection::General::Resonance)
// ------------------
// Builds one OneNucleusHybridPoint from a physical SpinAPI partition.
//
// Ownership contract:
//   * exactCoreSpins is ordered and defines the exact-core Kronecker basis;
//   * perturbativeNucleus is appended as the final product-space factor;
//   * every physical SpinSystem interaction is owned exactly once by either
//     exactCoreInteractions, perturbativeHyperfine, or
//     perturbativeNuclearInteractions;
//   * field-interaction lists are subsets used only for dH/dB;
//   * detection terms are explicit and use ResonanceMagneticMomentBuilder;
//   * SpinAPI owns all tensor algebra, frames, prefactors and embedding.
//
// Initial R2E-B scope is deliberately one perturbative nucleus.  Multi-nuclear
// convolution and automatic exact/perturbative partition selection are later
// layers.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_Resonance_HybridNuclearResonancePreparation
#define MOD_RunSection_General_Resonance_HybridNuclearResonancePreparation

#include "GeneralResonanceHamiltonian.h"
#include "HSExecutionPlan.h"
#include "HybridNuclearResonanceSolver.h"
#include "Interaction.h"
#include "ResonanceMagneticMomentBuilder.h"
#include "Spin.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "State.h"

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace RunSection::General::Resonance
{
    struct OneNucleusHybridPartition
    {
        SpinAPI::system_ptr system;

        // Ordered exact-core Hilbert factors.
        std::vector<SpinAPI::spin_ptr> exactCoreSpins;

        // Complete interaction ownership for the exact core.
        std::vector<SpinAPI::interaction_ptr> exactCoreInteractions;

        // Subset of exactCoreInteractions that scales with the swept field.
        std::vector<SpinAPI::interaction_ptr> exactCoreFieldInteractions;

        // Explicit EPR detection ownership.
        std::vector<ResonanceMagneticMomentTerm> detectionTerms;

        // Exactly one perturbative nuclear Hilbert factor.
        SpinAPI::spin_ptr perturbativeNucleus;

        // Cross-sector coupling.
        SpinAPI::interaction_ptr perturbativeHyperfine;

        // Complete one-nucleus interaction ownership.
        std::vector<SpinAPI::interaction_ptr>
            perturbativeNuclearInteractions;

        // Subset that scales with field.
        std::vector<SpinAPI::interaction_ptr>
            perturbativeNuclearFieldInteractions;

        // The perturbative nucleus is intentionally absent: the solver owns
        // an unpolarized 1/n reference density for that nucleus.
        SpinAPI::state_ptr exactCoreState;

        bool fullTensorRotation = true;
        double overlapThreshold = 1.0e-14;
        bool fieldIndependentProjection = false;
    };

    class HybridNuclearResonancePreparation
    {
    private:
        template <typename T>
        static bool Contains(
            const std::vector<std::shared_ptr<T>> &values,
            const std::shared_ptr<T> &value)
        {
            return std::find(
                values.begin(),values.end(),value) !=
                values.end();
        }

        template <typename T>
        static bool Unique(
            const std::vector<std::shared_ptr<T>> &values)
        {
            for (std::size_t i=0;i<values.size();++i)
            {
                if (values[i] == nullptr)
                    return false;
                for (std::size_t j=i+1;j<values.size();++j)
                {
                    if (values[i] == values[j])
                        return false;
                }
            }
            return true;
        }

        static bool AllSpinsIn(
            const std::vector<SpinAPI::spin_ptr> &spins,
            const std::vector<SpinAPI::spin_ptr> &allowed)
        {
            for (const auto &spin:spins)
            {
                if (spin == nullptr ||
                    !Contains(allowed,spin))
                    return false;
            }
            return true;
        }

        static bool InteractionUsesOnly(
            const SpinAPI::interaction_ptr &interaction,
            const std::vector<SpinAPI::spin_ptr> &allowed)
        {
            if (interaction == nullptr)
                return false;
            const auto g1=interaction->Group1();
            const auto g2=interaction->Group2();
            return !g1.empty() &&
                AllSpinsIn(g1,allowed) &&
                AllSpinsIn(g2,allowed);
        }

        static bool IsPerturbativeHyperfineOwnership(
            const SpinAPI::interaction_ptr &interaction,
            const SpinAPI::spin_ptr &nucleus,
            const std::vector<SpinAPI::spin_ptr> &core)
        {
            if (interaction == nullptr ||
                nucleus == nullptr ||
                interaction->Type() !=
                    SpinAPI::InteractionType::DoubleSpin)
                return false;

            const auto g1=interaction->Group1();
            const auto g2=interaction->Group2();

            const bool nucleusOnLeft=
                g1.size()==1 && g1.front()==nucleus &&
                !g2.empty() && AllSpinsIn(g2,core);
            const bool nucleusOnRight=
                g2.size()==1 && g2.front()==nucleus &&
                !g1.empty() && AllSpinsIn(g1,core);

            return nucleusOnLeft != nucleusOnRight;
        }

        static bool IsPerturbativeNuclearOwnership(
            const SpinAPI::interaction_ptr &interaction,
            const SpinAPI::spin_ptr &nucleus)
        {
            if (interaction == nullptr ||
                nucleus == nullptr)
                return false;

            const auto g1=interaction->Group1();
            const auto g2=interaction->Group2();

            if (g1.empty() || !g2.empty())
                return false;
            for (const auto &spin:g1)
            {
                if (spin != nucleus)
                    return false;
            }
            return true;
        }

        static int InteractionOwnershipCount(
            const OneNucleusHybridPartition &partition,
            const SpinAPI::interaction_ptr &interaction)
        {
            int count=0;
            for (const auto &owned:
                 partition.exactCoreInteractions)
            {
                if (owned==interaction)
                    ++count;
            }
            if (partition.perturbativeHyperfine==interaction)
                ++count;
            for (const auto &owned:
                 partition.perturbativeNuclearInteractions)
            {
                if (owned==interaction)
                    ++count;
            }
            return count;
        }

        static bool ValidateFieldInteraction(
            const SpinAPI::interaction_ptr &interaction,
            double fieldT,
            std::string &error,
            const std::string &label)
        {
            if (interaction == nullptr ||
                interaction->Type() !=
                    SpinAPI::InteractionType::SingleSpin)
            {
                error = label +
                    " field interaction must be a single-spin Zeeman interaction";
                return false;
            }
            if (interaction->HasTimeDependence())
            {
                error = label +
                    " field interaction must be static";
                return false;
            }

            const arma::vec field=interaction->Field();
            if (field.n_elem != 3 ||
                !field.is_finite())
            {
                error = label +
                    " field interaction must have a finite 3-vector";
                return false;
            }

            const double magnitude=arma::norm(field,2);
            const double tolerance=
                1.0e-10*std::max(1.0,std::abs(fieldT));
            if (!std::isfinite(magnitude) ||
                std::abs(magnitude-fieldT)>tolerance)
            {
                error = label +
                    " field interaction magnitude does not match the requested field";
                return false;
            }
            return true;
        }

        static HS::HSExecutionPlan CorePlan(
            const std::vector<SpinAPI::interaction_ptr> &interactions)
        {
            HS::HSExecutionPlan plan;
            plan.dynamics=HS::Dynamics::Static;
            plan.calculation=HS::Calculation::TimeEvolution;
            plan.sampling=HS::Sampling::Direct;
            plan.orientation=HS::OrientationMode::Explicit;
            plan.approximation=
                SpinAPI::HamiltonianApproximation::Full;
            plan.hasH0List=true;
            for (const auto &interaction:interactions)
                plan.h0List.push_back(interaction->Name());
            return plan;
        }

        static std::vector<std::string> Names(
            const std::vector<SpinAPI::interaction_ptr> &interactions)
        {
            std::vector<std::string> names;
            names.reserve(interactions.size());
            for (const auto &interaction:interactions)
                names.push_back(interaction->Name());
            return names;
        }

    public:
        static bool BuildPoint(
            const OneNucleusHybridPartition &partition,
            const HS::HSOrientation &orientation,
            double fieldT,
            OneNucleusHybridPoint &point,
            std::string &error)
        {
            error.clear();
            point=OneNucleusHybridPoint{};

            if (partition.system == nullptr)
            {
                error = "hybrid partition requires a SpinSystem";
                return false;
            }
            if (orientation.frameToLab.n_rows != 3 ||
                orientation.frameToLab.n_cols != 3 ||
                !orientation.frameToLab.is_finite())
            {
                error =
                    "hybrid partition orientation must be a finite 3x3 matrix";
                return false;
            }
            if (!std::isfinite(fieldT) || fieldT<=0.0)
            {
                error =
                    "hybrid partition field magnitude must be finite and positive";
                return false;
            }
            if (!std::isfinite(partition.overlapThreshold) ||
                partition.overlapThreshold<0.0 ||
                partition.overlapThreshold>1.0)
            {
                error =
                    "hybrid partition overlap threshold is invalid";
                return false;
            }

            if (partition.exactCoreSpins.empty() ||
                !Unique(partition.exactCoreSpins))
            {
                error =
                    "hybrid partition exact-core spins must be non-empty and unique";
                return false;
            }
            if (partition.perturbativeNucleus == nullptr ||
                !partition.system->Contains(
                    partition.perturbativeNucleus))
            {
                error =
                    "hybrid partition perturbative nucleus is missing from the SpinSystem";
                return false;
            }
            if (partition.perturbativeNucleus->Type() !=
                SpinAPI::SpinType::Nucleus)
            {
                error =
                    "hybrid partition perturbative spin must have nuclear SpinType";
                return false;
            }
            if (Contains(
                    partition.exactCoreSpins,
                    partition.perturbativeNucleus))
            {
                error =
                    "hybrid partition perturbative nucleus must not be part of the exact core";
                return false;
            }

            for (const auto &spin:partition.exactCoreSpins)
            {
                if (!partition.system->Contains(spin))
                {
                    error =
                        "hybrid partition exact-core spin is missing from the SpinSystem";
                    return false;
                }
            }

            const auto systemSpins=partition.system->Spins();
            if (systemSpins.size() !=
                partition.exactCoreSpins.size()+1)
            {
                error =
                    "hybrid partition must own every SpinSystem spin exactly once";
                return false;
            }
            for (const auto &spin:systemSpins)
            {
                const int count=
                    (Contains(partition.exactCoreSpins,spin)?1:0) +
                    (spin==partition.perturbativeNucleus?1:0);
                if (count!=1)
                {
                    error =
                        "hybrid partition must own every SpinSystem spin exactly once";
                    return false;
                }
            }

            if (partition.exactCoreInteractions.empty() ||
                !Unique(partition.exactCoreInteractions) ||
                !Unique(partition.perturbativeNuclearInteractions) ||
                partition.perturbativeHyperfine == nullptr)
            {
                error =
                    "hybrid partition interaction ownership is incomplete or duplicated";
                return false;
            }

            for (const auto &interaction:
                 partition.exactCoreInteractions)
            {
                if (!partition.system->Contains(interaction) ||
                    interaction->HasTimeDependence() ||
                    !InteractionUsesOnly(
                        interaction,
                        partition.exactCoreSpins))
                {
                    error =
                        "hybrid exact-core interaction is invalid or crosses the partition";
                    return false;
                }
            }

            if (!partition.system->Contains(
                    partition.perturbativeHyperfine) ||
                partition.perturbativeHyperfine->HasTimeDependence() ||
                !IsPerturbativeHyperfineOwnership(
                    partition.perturbativeHyperfine,
                    partition.perturbativeNucleus,
                    partition.exactCoreSpins))
            {
                error =
                    "hybrid perturbative hyperfine interaction does not connect nucleus and exact core exclusively";
                return false;
            }

            for (const auto &interaction:
                 partition.perturbativeNuclearInteractions)
            {
                if (!partition.system->Contains(interaction) ||
                    interaction->HasTimeDependence() ||
                    !IsPerturbativeNuclearOwnership(
                        interaction,
                        partition.perturbativeNucleus))
                {
                    error =
                        "hybrid perturbative nuclear interaction is invalid or crosses the partition";
                    return false;
                }
            }

            const auto systemInteractions=
                partition.system->Interactions();
            for (const auto &interaction:systemInteractions)
            {
                if (InteractionOwnershipCount(
                        partition,interaction)!=1)
                {
                    error =
                        "hybrid partition must own every SpinSystem interaction exactly once";
                    return false;
                }
            }

            if (partition.exactCoreFieldInteractions.empty() ||
                !Unique(partition.exactCoreFieldInteractions))
            {
                error =
                    "hybrid partition requires unique exact-core field interactions";
                return false;
            }
            for (const auto &interaction:
                 partition.exactCoreFieldInteractions)
            {
                if (!Contains(
                        partition.exactCoreInteractions,
                        interaction))
                {
                    error =
                        "hybrid exact-core field interaction is not owned by the exact core";
                    return false;
                }
                if (!ValidateFieldInteraction(
                        interaction,fieldT,error,
                        "hybrid exact-core"))
                    return false;
            }

            if (!Unique(
                    partition.perturbativeNuclearFieldInteractions))
            {
                error =
                    "hybrid perturbative nuclear field interactions must be unique";
                return false;
            }
            for (const auto &interaction:
                 partition.perturbativeNuclearFieldInteractions)
            {
                if (!Contains(
                        partition.perturbativeNuclearInteractions,
                        interaction))
                {
                    error =
                        "hybrid perturbative field interaction is not owned by the nucleus";
                    return false;
                }
                if (!ValidateFieldInteraction(
                        interaction,fieldT,error,
                        "hybrid perturbative nuclear"))
                    return false;
            }

            if (partition.detectionTerms.empty())
            {
                error =
                    "hybrid partition requires at least one EPR detection term";
                return false;
            }
            for (const auto &term:partition.detectionTerms)
            {
                if (term.spin==nullptr ||
                    term.zeeman==nullptr ||
                    !Contains(
                        partition.exactCoreSpins,
                        term.spin) ||
                    !Contains(
                        partition.exactCoreInteractions,
                        term.zeeman))
                {
                    error =
                        "hybrid detection term is not owned by the exact core";
                    return false;
                }
            }

            if (partition.exactCoreState == nullptr ||
                !partition.system->Contains(
                    partition.exactCoreState))
            {
                error =
                    "hybrid partition exact-core state is missing from the SpinSystem";
                return false;
            }

            std::vector<std::pair<int,arma::cx_double>>
                perturbativeStateSeries;
            bool perturbativeCoupled=false;
            if (partition.exactCoreState->GetStateSeries(
                    partition.perturbativeNucleus,
                    perturbativeStateSeries,
                    perturbativeCoupled) ||
                perturbativeCoupled)
            {
                error =
                    "hybrid perturbative nucleus reference state must be unpolarized and unspecified";
                return false;
            }

            SpinAPI::SpinSpace coreSpace(
                partition.exactCoreSpins);
            coreSpace.UseSuperoperatorSpace(false);
            coreSpace.UseFullTensorRotation(
                partition.fullTensorRotation);
            if (!coreSpace.Add(
                    partition.exactCoreInteractions))
            {
                error =
                    "failed to add exact-core interactions to SpinSpace";
                return false;
            }

            const auto corePlan=
                CorePlan(partition.exactCoreInteractions);
            GeneralResonanceHamiltonian coreBuilder(
                corePlan,coreSpace);

            if (!coreBuilder.Build(
                    orientation,
                    point.coreHamiltonian,error) ||
                !coreBuilder.BuildFieldDerivative(
                    orientation,
                    Names(partition.exactCoreFieldInteractions),
                    fieldT,
                    point.coreDHdB,error))
                return false;

            if (!coreSpace.GetState(
                    partition.exactCoreState,
                    point.coreDensity))
            {
                error =
                    "failed to construct the exact-core density matrix";
                return false;
            }
            const arma::cx_double trace=
                arma::trace(point.coreDensity);
            if (!std::isfinite(trace.real()) ||
                !std::isfinite(trace.imag()) ||
                std::abs(trace)<1.0e-15)
            {
                error =
                    "hybrid exact-core density matrix has invalid trace";
                return false;
            }
            point.coreDensity/=trace;
            if (!point.coreDensity.is_finite())
            {
                error =
                    "hybrid exact-core density matrix is non-finite";
                return false;
            }

            if (!ResonanceMagneticMomentBuilder::BuildTransverse(
                    coreSpace,
                    partition.detectionTerms,
                    orientation.frameToLab,
                    partition.fullTensorRotation,
                    point.coreMuX,
                    point.coreMuY,
                    error))
                return false;

            std::vector<SpinAPI::spin_ptr> pairSpins=
                partition.exactCoreSpins;
            pairSpins.push_back(
                partition.perturbativeNucleus);

            SpinAPI::SpinSpace pairSpace(pairSpins);
            pairSpace.UseSuperoperatorSpace(false);
            pairSpace.UseFullTensorRotation(
                partition.fullTensorRotation);

            arma::mat rotation=orientation.frameToLab;
            arma::sp_cx_mat hyperfinePair;
            if (!pairSpace.InteractionOperatorRotatedZYZ(
                    partition.perturbativeHyperfine,
                    rotation,
                    hyperfinePair))
            {
                error =
                    "failed to construct the exact-core/perturbative hyperfine operator";
                return false;
            }

            SpinAPI::SpinSpace nuclearSpace(
                partition.perturbativeNucleus);
            nuclearSpace.UseSuperoperatorSpace(false);
            nuclearSpace.UseFullTensorRotation(
                partition.fullTensorRotation);

            const arma::uword nuclearDimension=
                static_cast<arma::uword>(
                    partition.perturbativeNucleus->
                        Multiplicity());

            arma::sp_cx_mat nuclearHamiltonian(
                nuclearDimension,nuclearDimension);
            for (const auto &interaction:
                 partition.perturbativeNuclearInteractions)
            {
                arma::sp_cx_mat term;
                if (!nuclearSpace.InteractionOperatorRotatedZYZ(
                        interaction,rotation,term))
                {
                    error =
                        "failed to construct a perturbative nuclear interaction";
                    return false;
                }
                nuclearHamiltonian+=term;
            }

            arma::sp_cx_mat nuclearFieldHamiltonian(
                nuclearDimension,nuclearDimension);
            for (const auto &interaction:
                 partition.perturbativeNuclearFieldInteractions)
            {
                arma::sp_cx_mat term;
                if (!nuclearSpace.InteractionOperatorRotatedZYZ(
                        interaction,rotation,term))
                {
                    error =
                        "failed to construct a perturbative nuclear field interaction";
                    return false;
                }
                nuclearFieldHamiltonian+=term;
            }

            point.hybrid.hyperfineCoreNuclear=
                arma::cx_mat(hyperfinePair);
            point.hybrid.nuclearHamiltonian=
                arma::cx_mat(nuclearHamiltonian);
            point.hybrid.nuclearDHdB=
                nuclearFieldHamiltonian/fieldT;
            point.hybrid.nuclearDimension=
                nuclearDimension;
            point.hybrid.overlapThreshold=
                partition.overlapThreshold;
            point.hybrid.fieldIndependentProjection=
                partition.fieldIndependentProjection;

            return true;
        }
    };
}

#endif
