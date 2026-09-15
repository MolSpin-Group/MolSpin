/////////////////////////////////////////////////////////////////////////
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "ResonanceCalculations.h"
#include "ResonanceOrientationSampler.h"
#include "ResonanceSystemPreparation.h"
#include "ResonanceSpectrumEvaluator.h"
#include "HybridNuclearResonancePartitionBuilder.h"
#include "HybridNuclearResonancePreparation.h"
#include "HybridNuclearResonanceSolver.h"
#include "SpinSystem.h"
#include "Spin.h"
#include "Interaction.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>

namespace RunSection::General::Resonance
{
    bool ResonanceHybridCalculation::Point(const ResonanceExecutionPlan &plan,
                                           const SpinAPI::system_ptr &_system, ResonanceSample &sample,
                                           std::ostream &log, std::string &error, bool firstStep)
    {
        error.clear();

        if (_system == nullptr)
            return false;

        // The canonical hybrid partition owns every physical interaction exactly
        // once. Therefore a task-level H0 subset is valid only when it is the
        // complete static SpinSystem interaction set. We never silently add an
        // interaction omitted by the user and never silently drop physical terms.
        std::vector<std::string> h0list = plan.hamiltonianH0list;
        if (h0list.empty())
        {
            for (const auto &interaction : _system->Interactions())
            {
                if (interaction != nullptr && SpinAPI::IsStatic(*interaction))
                    h0list.push_back(interaction->Name());
            }
        }

        const auto systemInteractions = _system->Interactions();
        if (h0list.size() != systemInteractions.size())
        {
            log << "Hybrid resonance requires HamiltonianH0list to cover the complete static SpinSystem "
                   "interaction set exactly once."
                << std::endl;
            return false;
        }

        for (const auto &interaction : systemInteractions)
        {
            if (interaction == nullptr || interaction->HasTimeDependence())
            {
                log << "Hybrid resonance currently requires a completely static SpinSystem." << std::endl;
                return false;
            }

            const auto count =
                static_cast<std::size_t>(std::count(h0list.begin(), h0list.end(), interaction->Name()));
            if (count != 1)
            {
                log << "Hybrid resonance requires HamiltonianH0list to cover the complete static SpinSystem "
                       "interaction set exactly once."
                    << std::endl;
                return false;
            }
        }
        for (const auto &name : h0list)
        {
            if (_system->interactions_find(name) == nullptr ||
                std::count(h0list.begin(), h0list.end(), name) != 1)
            {
                log << "Hybrid resonance HamiltonianH0list contains an unknown or duplicate interaction."
                    << std::endl;
                return false;
            }
        }

        SpinAPI::interaction_ptr fieldInteraction = nullptr;
        if (!SystemPreparation::ResolveFieldInteraction(plan, _system, fieldInteraction) ||
            fieldInteraction == nullptr)
        {
            log << "Hybrid resonance requires a designated Zeeman field interaction." << std::endl;
            return false;
        }

        std::vector<SpinAPI::spin_ptr> detectSpins;
        std::vector<std::string> detectSpinNames;
        if (!SystemPreparation::ResolveDetectionSpins(plan, _system, fieldInteraction, detectSpins,
                                                      detectSpinNames) ||
            detectSpins.empty())
        {
            log << "Hybrid resonance failed to resolve exact-core EPR detection spins." << std::endl;
            return false;
        }

        std::vector<SpinAPI::interaction_ptr> zeemanInteractions =
            SystemPreparation::CollectZeemanInteractions(_system, h0list);
        if (zeemanInteractions.empty() || std::find(zeemanInteractions.begin(), zeemanInteractions.end(),
                                                    fieldInteraction) == zeemanInteractions.end())
        {
            log << "Hybrid resonance requires the designated field interaction and every physical Zeeman "
                   "term in "
                   "HamiltonianH0list."
                << std::endl;
            return false;
        }

        arma::vec Bvec = fieldInteraction->Field();
        if (Bvec.n_elem != 3 || !Bvec.is_finite())
        {
            log << "Hybrid resonance field interaction does not provide a finite 3-vector." << std::endl;
            return false;
        }

        FieldSyncGuard centerSync;
        if (plan.enforceZeemanSync)
        {
            centerSync.Apply(zeemanInteractions, Bvec);
        }
        else
        {
            const double tol = 1.0e-10;
            for (const auto &interaction : zeemanInteractions)
            {
                if (interaction == nullptr)
                    return false;
                const arma::vec field = interaction->Field();
                if (field.n_elem != 3 || !field.is_finite() || arma::norm(field - Bvec) > tol)
                {
                    log << "Hybrid resonance requires synchronized Zeeman field vectors; use "
                           "enforce_zeeman_sync=true or supply identical fields."
                        << std::endl;
                    return false;
                }
            }
        }

        // Re-read after optional synchronization.
        Bvec = fieldInteraction->Field();
        const double Bmag = arma::norm(Bvec);
        if (!std::isfinite(Bmag) || Bmag <= 0.0 || plan.hybridFieldStepT <= 0.0 ||
            plan.hybridFieldStepT >= Bmag)
        {
            log << "Hybrid resonance field magnitude/finite-difference step is invalid." << std::endl;
            return false;
        }
        const arma::vec fieldDirection = Bvec / Bmag;
        const double field_mT = 1.0e3 * Bmag;

        // R2K-C qualifies two distinct exact-core state policies:
        //   1. a fixed explicit state, preserving R2K-B semantics;
        //   2. an orientation/field-dependent Thermal state generated from an
        //      explicitly exact-core-owned thermal Hamiltonian.
        // Perturbative nuclear factors remain maximally mixed in the first-order
        // independent-factor solver. Molecular-frame explicit-state rotation is
        // still a separate unqualified transformation and therefore fails closed.
        const SpinAPI::StateFrame initialStateFrame = _system->InitialStateFrame();
        HybridNuclearResonanceCoreStateMode coreStateMode =
            HybridNuclearResonanceCoreStateMode::ExplicitFixed;
        SpinAPI::state_ptr exactCoreState = nullptr;
        std::vector<SpinAPI::interaction_ptr> exactCoreThermalInteractions;
        double thermalTemperatureK = 300.0;

        if (initialStateFrame == SpinAPI::StateFrame::Fixed)
        {
            if (!plan.initialStateName.empty())
            {
                exactCoreState = _system->states_find(plan.initialStateName);
            }
            else
            {
                const auto initialStates = _system->InitialState();
                if (initialStates.size() == 1 && initialStates.front() != nullptr)
                    exactCoreState = initialStates.front();
            }
            if (exactCoreState == nullptr)
            {
                log << "Hybrid resonance fixed-state mode requires exactly one explicit non-Thermal initial "
                       "state."
                    << std::endl;
                return false;
            }
        }
        else if (initialStateFrame == SpinAPI::StateFrame::Eigen)
        {
            if (!plan.initialStateName.empty())
            {
                log << "Hybrid resonance frame=eigen requires Thermal to be specified through the SpinSystem "
                       "initialstate property."
                    << std::endl;
                return false;
            }

            const auto initialStates = _system->InitialState();
            if (initialStates.size() != 1 || initialStates.front() != nullptr)
            {
                log << "Hybrid resonance frame=eigen requires a single Thermal SpinSystem initial state."
                    << std::endl;
                return false;
            }

            const auto thermalNames = _system->ThermalHamiltonianList();
            if (thermalNames.empty())
            {
                log << "Hybrid resonance R2K-C requires an explicit non-empty thermalhamiltonian list for "
                       "frame=eigen."
                    << std::endl;
                return false;
            }
            for (const auto &name : thermalNames)
            {
                auto interaction = _system->interactions_find(name);
                if (interaction == nullptr)
                {
                    log << "Hybrid resonance thermal Hamiltonian interaction \"" << name
                        << "\" was not found in the SpinSystem." << std::endl;
                    return false;
                }
                exactCoreThermalInteractions.push_back(interaction);
            }

            thermalTemperatureK = _system->Temperature();
            if (!std::isfinite(thermalTemperatureK) || thermalTemperatureK <= 0.0)
            {
                log << "Hybrid resonance thermal temperature must be finite and positive." << std::endl;
                return false;
            }

            coreStateMode = HybridNuclearResonanceCoreStateMode::ThermalEigen;
            log << "Hybrid resonance initial state = thermal exact core at " << thermalTemperatureK
                << " K; perturbative nuclear reference = maximally mixed." << std::endl;
        }
        else
        {
            log << "Hybrid resonance molecular-frame explicit-state rotation is not yet qualified; use "
                   "frame=fixed or frame=eigen with Thermal."
                << std::endl;
            return false;
        }

        std::vector<HybridNuclearResonanceExplicitNucleus> perturbative;
        perturbative.reserve(plan.hybridPerturbativeNucleusNames.size());
        for (const auto &name : plan.hybridPerturbativeNucleusNames)
        {
            auto nucleus = _system->spins_find(name);
            if (nucleus == nullptr || nucleus->Type() != SpinAPI::SpinType::Nucleus)
            {
                log << "Hybrid resonance perturbative nucleus \"" << name
                    << "\" is missing or is not SpinType::Nucleus." << std::endl;
                return false;
            }
            HybridNuclearResonanceExplicitNucleus spec;
            spec.nucleus = nucleus;
            spec.overlapThreshold = plan.hybridOverlapThreshold;
            spec.fieldIndependentProjection = false;
            perturbative.push_back(std::move(spec));
        }

        std::vector<ResonanceMagneticMomentTerm> detectionTerms;
        detectionTerms.reserve(detectSpins.size());
        for (const auto &spin : detectSpins)
        {
            auto zeeman = SystemPreparation::FindZeemanForSpin(spin, zeemanInteractions);
            if (zeeman == nullptr && fieldInteraction != nullptr)
            {
                const auto group = fieldInteraction->Group1();
                if (std::find(group.begin(), group.end(), spin) != group.end())
                    zeeman = fieldInteraction;
            }
            if (zeeman == nullptr)
            {
                log << "Hybrid resonance detection spin \"" << spin->Name()
                    << "\" has no owned Zeeman interaction." << std::endl;
                return false;
            }
            detectionTerms.push_back({spin, zeeman});
        }

        HybridNuclearResonanceExplicitPartitionRequest partitionRequest;
        partitionRequest.system = _system;
        partitionRequest.perturbativeNuclei = perturbative;
        partitionRequest.fieldInteractions = zeemanInteractions;
        partitionRequest.detectionTerms = detectionTerms;
        partitionRequest.coreStateMode = coreStateMode;
        partitionRequest.exactCoreState = exactCoreState;
        partitionRequest.exactCoreThermalInteractions = exactCoreThermalInteractions;
        partitionRequest.thermalTemperatureK = thermalTemperatureK;
        partitionRequest.fullTensorRotation = plan.fullTensorRotation;
        partitionRequest.minimumCumulativeOverlapWeight = plan.hybridMinimumCumulativeOverlapWeight;
        partitionRequest.maximumComponentsPerCoreTransition = plan.hybridMaximumComponentsPerCoreTransition;
        // Finite-difference branch tracking is currently qualified only for
        // unmerged center components. Merging remains disabled in this task gate.
        partitionRequest.mergeFrequencyToleranceRadNs = 0.0;

        HybridNuclearResonancePartition partition;
        std::string hybridError;
        if (!HybridNuclearResonancePartitionBuilder::Build(partitionRequest, partition, hybridError))
        {
            log << "Hybrid resonance partition rejected: " << hybridError << std::endl;
            return false;
        }

        ResonanceOrientationGrid orientations;
        if (!OrientationSampling::Build(plan, _system, fieldInteraction, h0list, orientations, log, error))
            return false;
        const auto &grid = orientations.grid;
        const int numPoints = static_cast<int>(grid.size()), gammaPoints = orientations.gammaPoints;
        const double gammaWeight = orientations.gammaWeight;

        SpectrumRequest resonanceRequest;
        resonanceRequest.microwaveFrequencyGHz = plan.mwFrequencyGHz;
        resonanceRequest.linewidth_mT = std::abs(plan.linewidth_mT);
        resonanceRequest.lineshape =
            (plan.lineshape == "lorentzian") ? Lineshape::Lorentzian : Lineshape::Gaussian;
        resonanceRequest.populationThreshold = 1.0e-15;
        resonanceRequest.minimumSlope = 1.0e-15;
        resonanceRequest.maximumDBdOmega = 1.0e5;

        HybridNuclearResonanceFieldResponseRequest fieldResponse;
        fieldResponse.fieldT = Bmag;
        fieldResponse.fieldStepT = plan.hybridFieldStepT;
        fieldResponse.minimumCoreStateOverlap = plan.hybridMinimumCoreStateOverlap;
        fieldResponse.minimumNuclearStateOverlap = plan.hybridMinimumNuclearStateOverlap;
        fieldResponse.jacobianRelativeTolerance = plan.hybridJacobianRelativeTolerance;
        fieldResponse.jacobianAbsoluteTolerance = plan.hybridJacobianAbsoluteTolerance;

        const std::size_t spinCount = detectSpins.size();
        double total_x = 0.0;
        double total_y = 0.0;
        double total_perp = 0.0;
        double cross_x = 0.0;
        double cross_y = 0.0;
        std::vector<double> spin_x(spinCount, 0.0);
        std::vector<double> spin_y(spinCount, 0.0);
        std::vector<double> spin_perp(spinCount, 0.0);
        std::vector<double> spin_p(spinCount, 0.0);
        std::vector<double> spin_m(spinCount, 0.0);

        std::size_t maxProductNuclearDimension = 1;
        std::size_t maxLargestDiagonalizedNuclearDimension = 0;
        double maxDiscardedWeight = 0.0;
        bool anyPruning = false;

        // Sequential on purpose in R2K-B: the finite-difference provider
        // temporarily mutates and restores shared physical Zeeman fields. A later
        // gate may parallelize over cloned immutable field realizations.
        for (int gridIndex = 0; gridIndex < numPoints; ++gridIndex)
        {
            auto [theta, phi, solidWeight] = grid[gridIndex];
            for (int gammaIndex = 0; gammaIndex < gammaPoints; ++gammaIndex)
            {
                const double gamma = (gammaPoints > 1)
                                         ? 2.0 * arma::datum::pi * (static_cast<double>(gammaIndex) + 0.5) /
                                               static_cast<double>(gammaPoints)
                                         : 0.0;
                const double weight = solidWeight * gammaWeight;

                arma::mat rotation;
                double alphaForRotation = phi;
                double betaForRotation = theta;
                double gammaForRotation = gamma;
                if (!OrientationSampling::Rotation(alphaForRotation, betaForRotation, gammaForRotation,
                                                   rotation))
                    return false;

                General::HS::HSOrientation orientation;
                orientation.alpha = phi;
                orientation.beta = theta;
                orientation.gamma = gamma;
                orientation.weight = weight;
                orientation.frameToLab = rotation;

                HybridNuclearResonancePointProvider provider =
                    [&](double fieldT, HybridNuclearResonancePoint &point, std::string &error)
                {
                    const arma::vec displacedField = fieldDirection * fieldT;
                    FieldSyncGuard displacedSync;
                    displacedSync.Apply(zeemanInteractions, displacedField);
                    return HybridNuclearResonancePreparation::BuildPoint(partition, orientation, fieldT,
                                                                         point, error);
                };

                ResonanceLineSet resonanceLines;
                HybridNuclearResonanceReport report;
                if (!HybridNuclearResonanceSolver::GenerateFirstOrderFiniteDifference(
                        provider, fieldResponse, resonanceRequest, resonanceLines, report, hybridError))
                {
                    log << "Hybrid resonance finite-difference solver rejected orientation: " << hybridError
                        << std::endl;
                    return false;
                }

                SpectrumPoint resonancePoint;
                if (!ResonanceSpectrumEvaluator::Evaluate(resonanceLines, resonanceRequest, resonancePoint,
                                                          hybridError))
                {
                    log << "Hybrid resonance spectrum evaluation failed: " << hybridError << std::endl;
                    return false;
                }
                if (resonancePoint.channels.size() != spinCount)
                {
                    log << "Hybrid resonance resolved detection-channel cardinality changed." << std::endl;
                    return false;
                }

                total_x += weight * resonancePoint.totalX;
                total_y += weight * resonancePoint.totalY;
                total_perp += weight * resonancePoint.totalPerpendicular;
                cross_x += weight * resonancePoint.crossX;
                cross_y += weight * resonancePoint.crossY;
                for (std::size_t i = 0; i < spinCount; ++i)
                {
                    spin_x[i] += weight * resonancePoint.channels[i].x;
                    spin_y[i] += weight * resonancePoint.channels[i].y;
                    spin_perp[i] += weight * resonancePoint.channels[i].perpendicular;
                    spin_p[i] += weight * resonancePoint.channels[i].plus;
                    spin_m[i] += weight * resonancePoint.channels[i].minus;
                }

                maxProductNuclearDimension =
                    std::max(maxProductNuclearDimension, report.productNuclearDimension);
                maxLargestDiagonalizedNuclearDimension = std::max(maxLargestDiagonalizedNuclearDimension,
                                                                  report.largestDiagonalizedNuclearDimension);
                maxDiscardedWeight =
                    std::max(maxDiscardedWeight, report.maximumDiscardedNuclearWeightFraction);
                anyPruning = anyPruning || report.pruningApplied;
            }
        }

        if (firstStep)
        {
            log << "Hybrid resonance explicit partition: " << perturbative.size()
                << " perturbative nuclei; product nuclear dimension = " << maxProductNuclearDimension
                << "; largest nuclear diagonalization = " << maxLargestDiagonalizedNuclearDimension
                << "; max discarded nuclear weight = " << maxDiscardedWeight
                << "; pruning = " << (anyPruning ? "yes" : "no") << "." << std::endl;
        }

        sample = ResonanceSample();
        sample.field_mT = field_mT;
        sample.spin_names = detectSpinNames;
        sample.point.totalX = total_x;
        sample.point.totalY = total_y;
        sample.point.totalPerpendicular = total_perp;
        sample.point.crossX = cross_x;
        sample.point.crossY = cross_y;
        sample.point.channels.resize(spinCount);
        for (size_t i = 0; i < spinCount; ++i)
        {
            auto &c = sample.point.channels[i];
            c.x = spin_x[i];
            c.y = spin_y[i];
            c.perpendicular = spin_perp[i];
            c.plus = spin_p[i];
            c.minus = spin_m[i];
        }
        return true;
    }
} // namespace RunSection::General::Resonance
