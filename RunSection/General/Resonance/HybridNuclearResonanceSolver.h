/////////////////////////////////////////////////////////////////////////
// HybridNuclearResonanceSolver (RunSection::General::Resonance)
// ------------------
// Exact-core, factorized perturbative-nuclear resonance backend.
//
// Qualified development layers:
//   R2A:
//     * exact electronic/exact-core diagonalization;
//     * one perturbative nucleus with an unpolarized reference density;
//     * SpinAPI-generated hyperfine operator on core x nucleus;
//     * nuclear Zeeman/static one-nucleus Hamiltonian;
//     * first-order effective nuclear sub-Hamiltonians.
//   R2C:
//     * field-response-correct hybrid Jacobian from symmetric finite differences;
//     * overlap-based tracking of core and nuclear eigenstates;
//     * h versus h/2 derivative convergence;
//     * fail-closed behavior when state tracking/convergence is not qualified.
//
//   R2G-A:
//     * independent first-order composition of several perturbative nuclei;
//     * no dense product nuclear Hamiltonian;
//     * explicit pruning, component-cap, merging and discarded-weight accounting.
//   R2G-B:
//     * finite-difference field response for independent multi-nuclear branches;
//     * one common exact-core tracking map plus one conditional-state map per nucleus;
//     * h versus h/2 convergence before fieldJacobianQualified=true.
//
// The solver does NOT parse g/A/D tensors. Hamiltonian matrices are generated
// upstream through SpinAPI/GeneralResonanceHamiltonian so tensor frames, powder
// rotations, units and prefactors remain owned by SpinAPI.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_Resonance_HybridNuclearResonanceSolver
#define MOD_RunSection_General_Resonance_HybridNuclearResonanceSolver

#include "ResonanceFieldJacobian.h"
#include "HybridNuclearShiftKernel.h"
#include "ResonanceTransitionMoments.h"
#include "ResonanceTypes.h"

#include <armadillo>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace RunSection::General::Resonance
{
    struct HybridNuclearResonanceNucleus
    {
        arma::cx_mat hyperfineCoreNuclear;
        arma::cx_mat nuclearHamiltonian;
        arma::sp_cx_mat nuclearDHdB;
        arma::uword nuclearDimension = 0;
        double overlapThreshold = 1.0e-14;

        // Legacy first-order analytic qualification is meaningful only for a
        // one-factor point. General field response is qualified through the
        // finite-difference API.
        bool fieldIndependentProjection = false;
    };

    struct HybridNuclearResonanceRequest
    {
        // Ordered independent perturbative nuclear factors. One nucleus is
        // represented by a vector with size()==1; there is no separate API.
        std::vector<HybridNuclearResonanceNucleus> nuclei;

        double minimumCumulativeOverlapWeight = 0.0;
        std::size_t maximumComponentsPerCoreTransition = 65536;
        double mergeFrequencyToleranceRadNs = 0.0;
        HybridNuclearCompositionMode compositionMode =
            HybridNuclearCompositionMode::Auto;
        double compressionTolerance_mT = 0.0;
    };

    struct HybridNuclearResonanceReport
    {
        std::size_t nucleusCount = 0;
        std::size_t productNuclearDimension = 1;
        std::size_t largestDiagonalizedNuclearDimension = 0;
        std::size_t coreTransitions = 0;
        std::size_t generatedComponents = 0;
        std::size_t retainedComponents = 0;
        std::size_t mergedComponents = 0;
        std::size_t outputComponents = 0;
        std::size_t maximumIntermediateComponents = 0;
        std::size_t formalCartesianComponents = 0;
        bool formalCartesianOverflow = false;
        HybridNuclearCompositionMode compositionBackend =
            HybridNuclearCompositionMode::Explicit;
        std::vector<std::size_t> maximumComponentsAfterStage;
        double maximumConvolutionWeightError = 0.0;
        double maximumDiscardedNuclearWeightFraction = 0.0;
        bool pruningApplied = false;
        bool mergingApplied = false;
    };

    struct HybridNuclearResonancePoint
    {
        arma::sp_cx_mat coreHamiltonian;
        arma::cx_mat coreDensity;
        arma::sp_cx_mat coreDHdB;
        arma::cx_mat coreMuX;
        arma::cx_mat coreMuY;

        // Optional ordered decomposition of the total transverse detection
        // operators. If present, it must sum back to coreMuX/coreMuY.
        std::vector<ResonanceDetectionOperator> coreDetectionChannels;

        HybridNuclearResonanceRequest hybrid;
    };

    using HybridNuclearResonancePointProvider =
        std::function<bool(
            double,
            HybridNuclearResonancePoint &,
            std::string &)>;

    struct HybridNuclearResonanceFieldResponseRequest
    {
        double fieldT = 0.0;
        double fieldStepT = 1.0e-4;
        double minimumCoreStateOverlap = 0.90;
        double minimumNuclearStateOverlap = 0.90;
        double jacobianRelativeTolerance = 1.0e-4;
        double jacobianAbsoluteTolerance = 1.0e-5;
    };

    class HybridNuclearResonanceSolver
    {
    private:
        struct NuclearManifold
        {
            arma::vec energies;
            arma::cx_mat eigenvectors;
            arma::vec dEdB;
        };

        struct PointSolution
        {
            arma::vec coreEnergies;
            arma::cx_mat coreEigenvectors;
            arma::vec coreDEdB;
            arma::vec corePopulations;
            arma::cx_mat muXEigen;
            arma::cx_mat muYEigen;
            std::vector<ResonanceDetectionOperator>
                detectionChannelsEigen;
            std::vector<NuclearManifold> nuclear;
            arma::uword nuclearDimension = 0;
            double overlapThreshold = 0.0;
        };

        struct HybridComponent
        {
            ResonanceLine line;
            arma::uword lowerNuclear = 0;
            arma::uword upperNuclear = 0;
        };

        struct IndependentNucleusSolution
        {
            std::vector<NuclearManifold> nuclear;
            arma::uword dimension = 0;
            double overlapThreshold = 0.0;
        };

        struct MultiPartialComponent
        {
            double nuclearShift = 0.0;
            double overlapWeight = 1.0;
            std::vector<arma::uword> lowerNuclear;
            std::vector<arma::uword> upperNuclear;
        };

        struct IndependentMultiComponent
        {
            ResonanceLine line;
            std::vector<arma::uword> lowerNuclear;
            std::vector<arma::uword> upperNuclear;
        };

        struct IndependentMultiPointSolution
        {
            PointSolution core;
            std::vector<IndependentNucleusSolution> factors;
            std::size_t productNuclearDimension = 1;
            std::size_t largestDiagonalizedNuclearDimension = 0;
        };

        struct IndependentMultiTrackingMaps
        {
            std::vector<arma::uword> core;
            std::vector<
                std::vector<
                    std::vector<arma::uword>>> nuclear;
        };

        static bool PartialCoreExpectation(
            const arma::cx_mat &operatorCoreNuclear,
            const arma::cx_vec &coreState,
            arma::uword nuclearDimension,
            arma::cx_mat &out)
        {
            out.reset();
            const arma::uword coreDimension = coreState.n_elem;
            if (coreDimension == 0 || nuclearDimension == 0 ||
                operatorCoreNuclear.n_rows !=
                    coreDimension*nuclearDimension ||
                operatorCoreNuclear.n_cols !=
                    coreDimension*nuclearDimension)
                return false;

            out.zeros(nuclearDimension,nuclearDimension);

            for (arma::uword i=0; i<coreDimension; ++i)
            {
                for (arma::uword j=0; j<coreDimension; ++j)
                {
                    const arma::cx_double weight =
                        std::conj(coreState(i))*coreState(j);
                    if (std::abs(weight) == 0.0)
                        continue;

                    const arma::uword r0 = i*nuclearDimension;
                    const arma::uword c0 = j*nuclearDimension;
                    out += weight * operatorCoreNuclear.submat(
                        r0,c0,
                        r0+nuclearDimension-1,
                        c0+nuclearDimension-1);
                }
            }

            out = 0.5*(out + out.t());
            return out.is_finite();
        }

        static bool SolveIndependentNucleus(
            const HybridNuclearResonanceNucleus &request,
            const arma::cx_mat &coreEigenvectors,
            IndependentNucleusSolution &solution,
            std::string &error)
        {
            solution = IndependentNucleusSolution{};

            const arma::uword coreDimension =
                coreEigenvectors.n_rows;
            const arma::uword nuclearDimension =
                request.nuclearDimension;

            if (coreDimension == 0 ||
                coreEigenvectors.n_cols != coreDimension ||
                nuclearDimension < 2 ||
                request.nuclearHamiltonian.n_rows != nuclearDimension ||
                request.nuclearHamiltonian.n_cols != nuclearDimension ||
                request.nuclearDHdB.n_rows != nuclearDimension ||
                request.nuclearDHdB.n_cols != nuclearDimension ||
                request.hyperfineCoreNuclear.n_rows !=
                    coreDimension*nuclearDimension ||
                request.hyperfineCoreNuclear.n_cols !=
                    coreDimension*nuclearDimension)
            {
                error =
                    "invalid independent perturbative-nucleus dimensions";
                return false;
            }
            if (!coreEigenvectors.is_finite() ||
                !request.nuclearHamiltonian.is_finite() ||
                !request.hyperfineCoreNuclear.is_finite() ||
                !std::isfinite(request.overlapThreshold) ||
                request.overlapThreshold < 0.0 ||
                request.overlapThreshold > 1.0)
            {
                error =
                    "invalid independent perturbative-nucleus numerical input";
                return false;
            }

            solution.dimension = nuclearDimension;
            solution.overlapThreshold =
                request.overlapThreshold;
            solution.nuclear.resize(coreDimension);

            for (arma::uword a=0; a<coreDimension; ++a)
            {
                arma::cx_mat projected;
                if (!PartialCoreExpectation(
                        request.hyperfineCoreNuclear,
                        coreEigenvectors.col(a),
                        nuclearDimension,projected))
                {
                    error =
                        "failed to project an independent perturbative hyperfine operator";
                    return false;
                }

                arma::cx_mat effectiveNuclear =
                    request.nuclearHamiltonian + projected;
                effectiveNuclear =
                    0.5*(effectiveNuclear + effectiveNuclear.t());

                if (!arma::eig_sym(
                        solution.nuclear[a].energies,
                        solution.nuclear[a].eigenvectors,
                        effectiveNuclear))
                {
                    error =
                        "failed to diagonalize an independent perturbative nuclear Hamiltonian";
                    return false;
                }

                if (!ResonanceFieldJacobian::ResolveDegenerateSubspaces(
                        solution.nuclear[a].energies,
                        solution.nuclear[a].eigenvectors,
                        request.nuclearDHdB,
                        solution.nuclear[a].dEdB,error))
                    return false;
            }

            return true;
        }

        static bool ValidatePoint(const HybridNuclearResonancePoint &point,
            const SpectrumRequest &request, std::string &error)
        {
            if (point.hybrid.nuclei.size()!=1)
            {
                error =
                    "single-factor hybrid helper requires exactly one perturbative nucleus";
                return false;
            }

            const auto &nucleus =
                point.hybrid.nuclei.front();
            const arma::uword coreDimension =
                point.coreHamiltonian.n_rows;
            const arma::uword nuclearDimension =
                nucleus.nuclearDimension;

            if (coreDimension == 0 ||
                point.coreHamiltonian.n_cols != coreDimension)
            {
                error =
                    "hybrid core Hamiltonian must be non-empty and square";
                return false;
            }
            if (point.coreDensity.n_rows != coreDimension ||
                point.coreDensity.n_cols != coreDimension ||
                point.coreDHdB.n_rows != coreDimension ||
                point.coreDHdB.n_cols != coreDimension ||
                point.coreMuX.n_rows != coreDimension ||
                point.coreMuX.n_cols != coreDimension ||
                point.coreMuY.n_rows != coreDimension ||
                point.coreMuY.n_cols != coreDimension)
            {
                error = "hybrid core operator dimensions do not match";
                return false;
            }
            if (!point.coreDetectionChannels.empty())
            {
                arma::cx_mat sumX(
                    coreDimension,coreDimension,
                    arma::fill::zeros);
                arma::cx_mat sumY(
                    coreDimension,coreDimension,
                    arma::fill::zeros);

                for (const auto &channel:
                     point.coreDetectionChannels)
                {
                    if (channel.x.n_rows!=coreDimension ||
                        channel.x.n_cols!=coreDimension ||
                        channel.y.n_rows!=coreDimension ||
                        channel.y.n_cols!=coreDimension ||
                        !channel.x.is_finite() ||
                        !channel.y.is_finite())
                    {
                        error =
                            "hybrid resolved detection channel dimensions do not match the core";
                        return false;
                    }
                    sumX+=channel.x;
                    sumY+=channel.y;
                }

                const double scale=std::max({
                    1.0,
                    arma::norm(point.coreMuX,"fro"),
                    arma::norm(point.coreMuY,"fro"),
                    arma::norm(sumX,"fro"),
                    arma::norm(sumY,"fro")
                });

                if (arma::norm(
                        sumX-point.coreMuX,"fro")>
                        1.0e-12*scale ||
                    arma::norm(
                        sumY-point.coreMuY,"fro")>
                        1.0e-12*scale)
                {
                    error =
                        "hybrid resolved detection channels do not sum to total transverse operators";
                    return false;
                }
            }

            if (nuclearDimension < 2 ||
                nucleus.nuclearHamiltonian.n_rows != nuclearDimension ||
                nucleus.nuclearHamiltonian.n_cols != nuclearDimension ||
                nucleus.nuclearDHdB.n_rows != nuclearDimension ||
                nucleus.nuclearDHdB.n_cols != nuclearDimension ||
                nucleus.hyperfineCoreNuclear.n_rows !=
                    coreDimension*nuclearDimension ||
                nucleus.hyperfineCoreNuclear.n_cols !=
                    coreDimension*nuclearDimension)
            {
                error = "invalid one-nucleus hybrid dimensions";
                return false;
            }
            if (!point.coreDensity.is_finite() ||
                !nucleus.nuclearHamiltonian.is_finite() ||
                !nucleus.hyperfineCoreNuclear.is_finite() ||
                !std::isfinite(nucleus.overlapThreshold) ||
                nucleus.overlapThreshold < 0.0 ||
                nucleus.overlapThreshold > 1.0 ||
                request.populationThreshold < 0.0 ||
                request.minimumSlope < 0.0 ||
                request.maximumDBdOmega <= 0.0)
            {
                error = "invalid one-nucleus hybrid numerical input";
                return false;
            }
            return true;
        }

        static bool SolvePoint(const HybridNuclearResonancePoint &point,
            const SpectrumRequest &request, PointSolution &solution,
            std::string &error)
        {
            error.clear();
            solution = PointSolution{};

            if (!ValidatePoint(point,request,error))
                return false;

            if (!arma::eig_sym(solution.coreEnergies,
                    solution.coreEigenvectors,
                    arma::cx_mat(point.coreHamiltonian)))
            {
                error = "failed to diagonalize hybrid core Hamiltonian";
                return false;
            }

            if (!ResonanceFieldJacobian::ResolveDegenerateSubspaces(
                    solution.coreEnergies,
                    solution.coreEigenvectors,
                    point.coreDHdB,
                    solution.coreDEdB,error))
                return false;

            const arma::cx_mat coreDensityEigen =
                solution.coreEigenvectors.t()*
                point.coreDensity*
                solution.coreEigenvectors;
            solution.corePopulations =
                arma::real(coreDensityEigen.diag());
            if (!solution.corePopulations.is_finite())
            {
                error = "non-finite hybrid core populations";
                return false;
            }

            if (!ResonanceTransitionMoments::Transform(
                    solution.coreEigenvectors,
                    point.coreMuX,point.coreMuY,
                    solution.muXEigen,solution.muYEigen,error) ||
                !ResonanceTransitionMoments::TransformChannels(
                    solution.coreEigenvectors,
                    point.coreDetectionChannels,
                    solution.detectionChannelsEigen,error))
                return false;

            IndependentNucleusSolution nucleus;
            if (!SolveIndependentNucleus(
                    point.hybrid.nuclei.front(),
                    solution.coreEigenvectors,
                    nucleus,error))
                return false;

            solution.nuclear =
                std::move(nucleus.nuclear);
            solution.nuclearDimension =
                nucleus.dimension;
            solution.overlapThreshold =
                nucleus.overlapThreshold;

            return true;
        }

        static bool BuildComponents(const PointSolution &solution,
            const SpectrumRequest &request, bool applyAnalyticSlope,
            std::vector<HybridComponent> &components,
            std::string &error)
        {
            components.clear();
            const arma::uword coreDimension =
                solution.coreEnergies.n_elem;
            const arma::uword nuclearDimension =
                solution.nuclearDimension;
            const double invNuclearDimension =
                1.0/static_cast<double>(nuclearDimension);

            for (arma::uword lower=0; lower<coreDimension; ++lower)
            {
                for (arma::uword upper=lower+1;
                     upper<coreDimension; ++upper)
                {
                    const double corePopulationDifference =
                        solution.corePopulations(lower)-
                        solution.corePopulations(upper);
                    TransitionMoment coreMoment;
                    if (solution.detectionChannelsEigen.empty())
                    {
                        coreMoment =
                            ResonanceTransitionMoments::Evaluate(
                                solution.muXEigen,
                                solution.muYEigen,
                                lower,upper);
                    }
                    else if (!ResonanceTransitionMoments::EvaluateResolved(
                            solution.muXEigen,
                            solution.muYEigen,
                            solution.detectionChannelsEigen,
                            lower,upper,
                            coreMoment,error))
                    {
                        return false;
                    }

                    for (arma::uword r=0; r<nuclearDimension; ++r)
                    {
                        for (arma::uword s=0; s<nuclearDimension; ++s)
                        {
                            const arma::cx_double overlap =
                                arma::cdot(
                                    solution.nuclear[upper].
                                        eigenvectors.col(s),
                                    solution.nuclear[lower].
                                        eigenvectors.col(r));
                            const double overlapWeight =
                                std::norm(overlap);
                            if (overlapWeight <
                                solution.overlapThreshold)
                                continue;

                            const double populationDifference =
                                corePopulationDifference*
                                invNuclearDimension;
                            if (std::abs(populationDifference) <
                                request.populationThreshold)
                                continue;

                            HybridComponent component;
                            component.line.lower = lower;
                            component.line.upper = upper;
                            component.lowerNuclear = r;
                            component.upperNuclear = s;
                            component.line.omega =
                                solution.coreEnergies(upper)-
                                solution.coreEnergies(lower) +
                                solution.nuclear[upper].energies(s)-
                                solution.nuclear[lower].energies(r);
                            component.line.populationDifference =
                                populationDifference;
                            component.line.moment =
                                ResonanceTransitionMoments::Scale(
                                    coreMoment,overlapWeight);

                            if (applyAnalyticSlope)
                            {
                                const double signedSlope =
                                    (solution.coreDEdB(upper)-
                                     solution.coreDEdB(lower)) +
                                    (solution.nuclear[upper].dEdB(s)-
                                     solution.nuclear[lower].dEdB(r));
                                const double slope =
                                    std::abs(signedSlope);
                                if (!std::isfinite(slope) ||
                                    slope < request.minimumSlope)
                                    continue;

                                const double dBdOmega = 1.0/slope;
                                if (!std::isfinite(dBdOmega) ||
                                    dBdOmega >
                                    request.maximumDBdOmega)
                                    continue;

                                component.line.dOmegaDB = slope;
                                component.line.dBdOmega = dBdOmega;
                            }

                            components.push_back(component);
                        }
                    }
                }
            }
            return true;
        }

        static bool MatchEigenvectors(const arma::cx_mat &reference,
            const arma::cx_mat &displaced, double minimumOverlap,
            std::vector<arma::uword> &mapping, std::string &error,
            const std::string &label)
        {
            mapping.clear();
            if (reference.n_rows != displaced.n_rows ||
                reference.n_cols != displaced.n_cols ||
                reference.n_cols == 0)
            {
                error = "hybrid field-response " + label +
                    " dimensions do not match";
                return false;
            }

            const arma::uword n = reference.n_cols;
            mapping.resize(n);
            std::vector<bool> used(n,false);

            for (arma::uword i=0; i<n; ++i)
            {
                double best = -1.0;
                arma::uword bestIndex = 0;
                for (arma::uword j=0; j<n; ++j)
                {
                    const double overlap =
                        std::norm(arma::cdot(
                            reference.col(i),displaced.col(j)));
                    if (overlap > best)
                    {
                        best = overlap;
                        bestIndex = j;
                    }
                }

                if (!std::isfinite(best) ||
                    best < minimumOverlap)
                {
                    error = "hybrid field-response " + label +
                        " state overlap below threshold";
                    return false;
                }
                if (used[bestIndex])
                {
                    error = "hybrid field-response " + label +
                        " state mapping is ambiguous";
                    return false;
                }

                used[bestIndex] = true;
                mapping[i] = bestIndex;
            }

            return true;
        }

        static bool BuildTrackingMaps(const PointSolution &reference,
            const PointSolution &displaced,
            double minimumCoreOverlap,
            double minimumNuclearOverlap,
            std::vector<arma::uword> &coreMap,
            std::vector<std::vector<arma::uword>> &nuclearMap,
            std::string &error)
        {
            if (!MatchEigenvectors(
                    reference.coreEigenvectors,
                    displaced.coreEigenvectors,
                    minimumCoreOverlap,coreMap,error,"core"))
                return false;

            nuclearMap.clear();
            nuclearMap.resize(reference.coreEnergies.n_elem);

            for (arma::uword a=0;
                 a<reference.coreEnergies.n_elem; ++a)
            {
                const arma::uword displacedCore = coreMap[a];
                if (!MatchEigenvectors(
                        reference.nuclear[a].eigenvectors,
                        displaced.nuclear[displacedCore].eigenvectors,
                        minimumNuclearOverlap,
                        nuclearMap[a],error,"nuclear"))
                    return false;
            }

            return true;
        }

        static bool TrackedFrequency(const HybridComponent &component,
            const PointSolution &displaced,
            const std::vector<arma::uword> &coreMap,
            const std::vector<std::vector<arma::uword>> &nuclearMap,
            double &omega)
        {
            const arma::uword lower =
                coreMap[component.line.lower];
            const arma::uword upper =
                coreMap[component.line.upper];
            const arma::uword r =
                nuclearMap[component.line.lower]
                          [component.lowerNuclear];
            const arma::uword s =
                nuclearMap[component.line.upper]
                          [component.upperNuclear];

            omega =
                displaced.coreEnergies(upper)-
                displaced.coreEnergies(lower) +
                displaced.nuclear[upper].energies(s)-
                displaced.nuclear[lower].energies(r);

            return std::isfinite(omega);
        }

        static bool SolveTrackedDisplacement(
            const HybridNuclearResonancePointProvider &provider,
            double fieldT,
            const SpectrumRequest &request,
            const PointSolution &reference,
            double minimumCoreOverlap,
            double minimumNuclearOverlap,
            PointSolution &solution,
            std::vector<arma::uword> &coreMap,
            std::vector<std::vector<arma::uword>> &nuclearMap,
            std::string &error)
        {
            HybridNuclearResonancePoint point;
            if (!provider(fieldT,point,error))
            {
                if (error.empty())
                    error =
                        "hybrid field-response point provider failed";
                return false;
            }
            if (!SolvePoint(point,request,solution,error))
                return false;
            return BuildTrackingMaps(
                reference,solution,
                minimumCoreOverlap,minimumNuclearOverlap,
                coreMap,nuclearMap,error);
        }

        static bool ValidateIndependentMultiRequest(
            const HybridNuclearResonanceRequest &hybrid,
            std::string &error)
        {
            if (hybrid.nuclei.empty())
            {
                error =
                    "independent multi-nucleus hybrid requires at least one perturbative nucleus";
                return false;
            }
            if (!std::isfinite(
                    hybrid.minimumCumulativeOverlapWeight) ||
                hybrid.minimumCumulativeOverlapWeight < 0.0 ||
                hybrid.minimumCumulativeOverlapWeight > 1.0 ||
                !std::isfinite(
                    hybrid.mergeFrequencyToleranceRadNs) ||
                hybrid.mergeFrequencyToleranceRadNs < 0.0 ||
                !std::isfinite(hybrid.compressionTolerance_mT) ||
                hybrid.compressionTolerance_mT < 0.0)
            {
                error =
                    "invalid independent multi-nucleus composition request";
                return false;
            }
            return true;
        }

        static bool SolveIndependentMultiPoint(
            const HybridNuclearResonancePoint &point,
            const SpectrumRequest &request,
            IndependentMultiPointSolution &solution,
            std::string &error)
        {
            solution = IndependentMultiPointSolution{};

            if (!ValidateIndependentMultiRequest(
                    point.hybrid,error))
                return false;

            HybridNuclearResonancePoint firstPoint;
            firstPoint.coreHamiltonian =
                point.coreHamiltonian;
            firstPoint.coreDensity =
                point.coreDensity;
            firstPoint.coreDHdB =
                point.coreDHdB;
            firstPoint.coreMuX =
                point.coreMuX;
            firstPoint.coreMuY =
                point.coreMuY;
            firstPoint.coreDetectionChannels =
                point.coreDetectionChannels;
            firstPoint.hybrid.nuclei = {
                point.hybrid.nuclei.front()};

            if (!SolvePoint(
                    firstPoint,request,
                    solution.core,error))
                return false;

            solution.factors.reserve(
                point.hybrid.nuclei.size());

            IndependentNucleusSolution first;
            first.nuclear =
                solution.core.nuclear;
            first.dimension =
                solution.core.nuclearDimension;
            first.overlapThreshold =
                solution.core.overlapThreshold;
            solution.factors.push_back(
                std::move(first));

            for (std::size_t k=1;
                 k<point.hybrid.nuclei.size(); ++k)
            {
                IndependentNucleusSolution factor;
                if (!SolveIndependentNucleus(
                        point.hybrid.nuclei[k],
                        solution.core.coreEigenvectors,
                        factor,error))
                    return false;
                solution.factors.push_back(
                    std::move(factor));
            }

            solution.productNuclearDimension = 1;
            solution.largestDiagonalizedNuclearDimension = 0;

            for (const auto &factor:solution.factors)
            {
                const std::size_t dimension =
                    static_cast<std::size_t>(
                        factor.dimension);
                if (dimension == 0 ||
                    solution.productNuclearDimension >
                        std::numeric_limits<std::size_t>::max()/
                            dimension)
                {
                    error =
                        "independent multi-nucleus product dimension overflow";
                    return false;
                }

                solution.productNuclearDimension *=
                    dimension;
                solution.largestDiagonalizedNuclearDimension =
                    std::max(
                        solution.
                            largestDiagonalizedNuclearDimension,
                        dimension);
            }

            return true;
        }

        static bool BuildIndependentMultiComponents(
            const IndependentMultiPointSolution &solution,
            const HybridNuclearResonanceRequest &hybrid,
            const SpectrumRequest &request,
            std::vector<IndependentMultiComponent> &components,
            HybridNuclearResonanceReport &report,
            std::string &error)
        {
            components.clear();
            report = HybridNuclearResonanceReport{};
            report.nucleusCount =
                solution.factors.size();
            report.productNuclearDimension =
                solution.productNuclearDimension;
            report.largestDiagonalizedNuclearDimension =
                solution.largestDiagonalizedNuclearDimension;

            if (report.nucleusCount == 0 ||
                report.productNuclearDimension == 0)
            {
                error =
                    "independent multi-nucleus solution is empty";
                return false;
            }

            const double productDimension =
                static_cast<double>(
                    report.productNuclearDimension);
            const arma::uword coreDimension =
                solution.core.coreEnergies.n_elem;

            for (arma::uword lower=0;
                 lower<coreDimension; ++lower)
            {
                for (arma::uword upper=lower+1;
                     upper<coreDimension; ++upper)
                {
                    ++report.coreTransitions;

                    const double corePopulationDifference =
                        solution.core.corePopulations(lower)-
                        solution.core.corePopulations(upper);
                    const double populationDifference =
                        corePopulationDifference/
                        productDimension;

                    if (std::abs(populationDifference) <
                        request.populationThreshold)
                        continue;

                    TransitionMoment coreMoment;
                    if (solution.core.
                            detectionChannelsEigen.empty())
                    {
                        coreMoment =
                            ResonanceTransitionMoments::Evaluate(
                                solution.core.muXEigen,
                                solution.core.muYEigen,
                                lower,upper);
                    }
                    else if (!ResonanceTransitionMoments::EvaluateResolved(
                            solution.core.muXEigen,
                            solution.core.muYEigen,
                            solution.core.detectionChannelsEigen,
                            lower,upper,
                            coreMoment,error))
                    {
                        return false;
                    }

                    std::vector<MultiPartialComponent>
                        partials(1);

                    for (const auto &factor:
                         solution.factors)
                    {
                        std::vector<MultiPartialComponent>
                            next;
                        const std::size_t componentLimit=
                            hybrid.maximumComponentsPerCoreTransition==0
                            ? 65536
                            : hybrid.maximumComponentsPerCoreTransition;
                        const std::size_t dimension=
                            static_cast<std::size_t>(factor.dimension);
                        if (dimension==0 || dimension>
                            std::numeric_limits<std::size_t>::max()/dimension)
                        {
                            error =
                                "independent multi-nucleus component count overflow";
                            return false;
                        }
                        const std::size_t localComponents=
                            dimension*dimension;
                        if (partials.size()>
                                std::numeric_limits<std::size_t>::max()/
                                    localComponents ||
                            partials.size()*localComponents>
                                componentLimit)
                        {
                            error =
                                "independent multi-nucleus component cap exceeded";
                            return false;
                        }
                        next.reserve(
                            partials.size()*localComponents);
                        const auto &lowerManifold =
                            factor.nuclear[lower];
                        const auto &upperManifold =
                            factor.nuclear[upper];

                        for (const auto &partial:partials)
                        {
                            for (arma::uword r=0;
                                 r<factor.dimension; ++r)
                            {
                                for (arma::uword s=0;
                                     s<factor.dimension; ++s)
                                {
                                    ++report.generatedComponents;

                                    const arma::cx_double overlap =
                                        arma::cdot(
                                            upperManifold.
                                                eigenvectors.col(s),
                                            lowerManifold.
                                                eigenvectors.col(r));
                                    double overlapWeight =
                                        std::norm(overlap);

                                    if (!std::isfinite(
                                            overlapWeight) ||
                                        overlapWeight < 0.0 ||
                                        overlapWeight >
                                            1.0+1.0e-10)
                                    {
                                        error =
                                            "invalid independent nuclear-state overlap weight";
                                        return false;
                                    }
                                    overlapWeight =
                                        std::min(
                                            1.0,overlapWeight);

                                    if (overlapWeight <
                                        factor.overlapThreshold)
                                    {
                                        report.pruningApplied =
                                            true;
                                        continue;
                                    }

                                    const double cumulativeWeight =
                                        partial.overlapWeight*
                                        overlapWeight;
                                    if (cumulativeWeight <
                                        hybrid.
                                            minimumCumulativeOverlapWeight)
                                    {
                                        report.pruningApplied =
                                            true;
                                        continue;
                                    }

                                    MultiPartialComponent component =
                                        partial;
                                    component.nuclearShift +=
                                        upperManifold.energies(s)-
                                        lowerManifold.energies(r);
                                    component.overlapWeight =
                                        cumulativeWeight;
                                    component.lowerNuclear.
                                        push_back(r);
                                    component.upperNuclear.
                                        push_back(s);

                                    if (!std::isfinite(
                                            component.
                                                nuclearShift) ||
                                        !std::isfinite(
                                            component.
                                                overlapWeight))
                                    {
                                        error =
                                            "non-finite independent multi-nucleus component";
                                        return false;
                                    }

                                    next.push_back(
                                        std::move(component));

                                    if (hybrid.
                                            maximumComponentsPerCoreTransition >
                                            0 &&
                                        next.size() >
                                            hybrid.
                                                maximumComponentsPerCoreTransition)
                                    {
                                        error =
                                            "independent multi-nucleus component cap exceeded";
                                        return false;
                                    }
                                }
                            }
                        }

                        partials.swap(next);
                        report.maximumIntermediateComponents =
                            std::max(
                                report.
                                    maximumIntermediateComponents,
                                partials.size());

                        if (partials.empty())
                            break;
                    }

                    double retainedWeight = 0.0;
                    for (const auto &partial:partials)
                        retainedWeight +=
                            partial.overlapWeight;

                    if (!std::isfinite(retainedWeight))
                    {
                        error =
                            "non-finite retained independent nuclear weight";
                        return false;
                    }

                    double retainedFraction =
                        retainedWeight/
                        productDimension;
                    if (retainedFraction >
                        1.0+1.0e-10)
                    {
                        error =
                            "independent multi-nucleus overlap weight is not normalized";
                        return false;
                    }
                    retainedFraction =
                        std::max(
                            0.0,
                            std::min(
                                1.0,retainedFraction));
                    report.
                        maximumDiscardedNuclearWeightFraction =
                        std::max(
                            report.
                                maximumDiscardedNuclearWeightFraction,
                            1.0-retainedFraction);

                    report.retainedComponents +=
                        partials.size();

                    if (hybrid.
                            mergeFrequencyToleranceRadNs >
                            0.0 &&
                        partials.size() > 1)
                    {
                        std::sort(
                            partials.begin(),
                            partials.end(),
                            [](const MultiPartialComponent &a,
                               const MultiPartialComponent &b)
                            {
                                if (a.nuclearShift !=
                                    b.nuclearShift)
                                    return
                                        a.nuclearShift <
                                        b.nuclearShift;
                                return
                                    a.overlapWeight <
                                    b.overlapWeight;
                            });

                        std::vector<MultiPartialComponent>
                            merged;
                        merged.reserve(partials.size());

                        for (const auto &partial:partials)
                        {
                            if (merged.empty() ||
                                std::abs(
                                    partial.nuclearShift-
                                    merged.back().
                                        nuclearShift) >
                                    hybrid.
                                        mergeFrequencyToleranceRadNs)
                            {
                                merged.push_back(partial);
                                continue;
                            }

                            const double combinedWeight =
                                merged.back().
                                    overlapWeight+
                                partial.overlapWeight;
                            if (!(combinedWeight>0.0) ||
                                !std::isfinite(
                                    combinedWeight))
                            {
                                error =
                                    "invalid independent multi-nucleus merged weight";
                                return false;
                            }

                            merged.back().nuclearShift =
                                (merged.back().
                                     nuclearShift*
                                     merged.back().
                                         overlapWeight +
                                 partial.nuclearShift*
                                     partial.overlapWeight)/
                                combinedWeight;
                            merged.back().overlapWeight =
                                combinedWeight;
                        }

                        if (merged.size() <
                            partials.size())
                        {
                            report.mergingApplied = true;
                            report.mergedComponents +=
                                partials.size()-
                                merged.size();
                        }
                        partials.swap(merged);
                    }

                    const double coreGap =
                        solution.core.
                            coreEnergies(upper)-
                        solution.core.
                            coreEnergies(lower);

                    for (const auto &partial:partials)
                    {
                        IndependentMultiComponent output;
                        output.line.lower = lower;
                        output.line.upper = upper;
                        output.line.omega =
                            coreGap+
                            partial.nuclearShift;
                        output.line.populationDifference =
                            populationDifference;
                        output.line.moment =
                            ResonanceTransitionMoments::Scale(
                                coreMoment,
                                partial.overlapWeight);
                        output.lowerNuclear =
                            partial.lowerNuclear;
                        output.upperNuclear =
                            partial.upperNuclear;

                        if (!std::isfinite(
                                output.line.omega) ||
                            !std::isfinite(
                                output.line.
                                    populationDifference) ||
                            !ResonanceTransitionMoments::IsFinite(
                                output.line.moment))
                        {
                            error =
                                "non-finite independent multi-nucleus resonance line";
                            return false;
                        }

                        components.push_back(
                            std::move(output));
                    }
                }
            }

            report.outputComponents =
                components.size();
            return true;
        }

        static bool BuildIndependentMultiTrackingMaps(
            const IndependentMultiPointSolution &reference,
            const IndependentMultiPointSolution &displaced,
            double minimumCoreOverlap,
            double minimumNuclearOverlap,
            IndependentMultiTrackingMaps &maps,
            std::string &error)
        {
            maps = IndependentMultiTrackingMaps{};

            if (reference.factors.size() !=
                displaced.factors.size())
            {
                error =
                    "multi-nucleus field-response nucleus count changed";
                return false;
            }

            if (!MatchEigenvectors(
                    reference.core.coreEigenvectors,
                    displaced.core.coreEigenvectors,
                    minimumCoreOverlap,
                    maps.core,error,
                    "multi-nucleus core"))
                return false;

            maps.nuclear.resize(
                reference.factors.size());

            for (std::size_t k=0;
                 k<reference.factors.size(); ++k)
            {
                if (reference.factors[k].dimension !=
                    displaced.factors[k].dimension)
                {
                    error =
                        "multi-nucleus field-response nuclear dimension changed";
                    return false;
                }

                maps.nuclear[k].resize(
                    reference.core.
                        coreEnergies.n_elem);

                for (arma::uword a=0;
                     a<reference.core.
                         coreEnergies.n_elem; ++a)
                {
                    const arma::uword displacedCore =
                        maps.core[a];
                    if (!MatchEigenvectors(
                            reference.factors[k].
                                nuclear[a].eigenvectors,
                            displaced.factors[k].
                                nuclear[displacedCore].
                                    eigenvectors,
                            minimumNuclearOverlap,
                            maps.nuclear[k][a],
                            error,
                            "multi-nucleus conditional nucleus"))
                        return false;
                }
            }

            return true;
        }

        static bool TrackedIndependentMultiFrequency(
            const IndependentMultiComponent &component,
            const IndependentMultiPointSolution &displaced,
            const IndependentMultiTrackingMaps &maps,
            double &omega)
        {
            if (component.lowerNuclear.size() !=
                    displaced.factors.size() ||
                component.upperNuclear.size() !=
                    displaced.factors.size() ||
                maps.nuclear.size() !=
                    displaced.factors.size() ||
                component.line.lower >=
                    maps.core.size() ||
                component.line.upper >=
                    maps.core.size())
                return false;

            const arma::uword lower =
                maps.core[component.line.lower];
            const arma::uword upper =
                maps.core[component.line.upper];

            omega =
                displaced.core.coreEnergies(upper)-
                displaced.core.coreEnergies(lower);

            for (std::size_t k=0;
                 k<displaced.factors.size(); ++k)
            {
                if (component.line.lower >=
                        maps.nuclear[k].size() ||
                    component.line.upper >=
                        maps.nuclear[k].size() ||
                    component.lowerNuclear[k] >=
                        maps.nuclear[k]
                            [component.line.lower].size() ||
                    component.upperNuclear[k] >=
                        maps.nuclear[k]
                            [component.line.upper].size())
                    return false;

                const arma::uword r =
                    maps.nuclear[k]
                        [component.line.lower]
                        [component.lowerNuclear[k]];
                const arma::uword s =
                    maps.nuclear[k]
                        [component.line.upper]
                        [component.upperNuclear[k]];

                omega +=
                    displaced.factors[k].
                        nuclear[upper].energies(s)-
                    displaced.factors[k].
                        nuclear[lower].energies(r);
            }

            return std::isfinite(omega);
        }

        static bool TrackedIndependentCoreGap(
            arma::uword lowerReference,
            arma::uword upperReference,
            const IndependentMultiPointSolution &displaced,
            const IndependentMultiTrackingMaps &maps,
            double &gap)
        {
            if (lowerReference >= maps.core.size() ||
                upperReference >= maps.core.size())
                return false;
            const arma::uword lower=maps.core[lowerReference];
            const arma::uword upper=maps.core[upperReference];
            if (lower >= displaced.core.coreEnergies.n_elem ||
                upper >= displaced.core.coreEnergies.n_elem)
                return false;
            gap=displaced.core.coreEnergies(upper)-
                displaced.core.coreEnergies(lower);
            return std::isfinite(gap);
        }

        static bool TrackedIndependentNuclearShift(
            std::size_t factorIndex,
            arma::uword lowerReference,
            arma::uword upperReference,
            arma::uword lowerNuclearReference,
            arma::uword upperNuclearReference,
            const IndependentMultiPointSolution &displaced,
            const IndependentMultiTrackingMaps &maps,
            double &shift)
        {
            if (factorIndex >= displaced.factors.size() ||
                factorIndex >= maps.nuclear.size() ||
                lowerReference >= maps.core.size() ||
                upperReference >= maps.core.size() ||
                lowerReference >= maps.nuclear[factorIndex].size() ||
                upperReference >= maps.nuclear[factorIndex].size() ||
                lowerNuclearReference >=
                    maps.nuclear[factorIndex][lowerReference].size() ||
                upperNuclearReference >=
                    maps.nuclear[factorIndex][upperReference].size())
                return false;

            const arma::uword lower=maps.core[lowerReference];
            const arma::uword upper=maps.core[upperReference];
            const arma::uword r=
                maps.nuclear[factorIndex][lowerReference]
                    [lowerNuclearReference];
            const arma::uword s=
                maps.nuclear[factorIndex][upperReference]
                    [upperNuclearReference];
            if (lower >= displaced.factors[factorIndex].nuclear.size() ||
                upper >= displaced.factors[factorIndex].nuclear.size() ||
                r >= displaced.factors[factorIndex].nuclear[lower].energies.n_elem ||
                s >= displaced.factors[factorIndex].nuclear[upper].energies.n_elem)
                return false;

            shift=
                displaced.factors[factorIndex].nuclear[upper].energies(s)-
                displaced.factors[factorIndex].nuclear[lower].energies(r);
            return std::isfinite(shift);
        }

        static bool SolveTrackedIndependentMultiDisplacement(
            const HybridNuclearResonancePointProvider &provider,
            double fieldT,
            const SpectrumRequest &request,
            const IndependentMultiPointSolution &reference,
            double minimumCoreOverlap,
            double minimumNuclearOverlap,
            IndependentMultiPointSolution &solution,
            IndependentMultiTrackingMaps &maps,
            std::string &error)
        {
            HybridNuclearResonancePoint point;
            if (!provider(fieldT,point,error))
            {
                if (error.empty())
                    error =
                        "multi-nucleus field-response point provider failed";
                return false;
            }

            if (!SolveIndependentMultiPoint(
                    point,request,solution,error))
                return false;

            return BuildIndependentMultiTrackingMaps(
                reference,solution,
                minimumCoreOverlap,
                minimumNuclearOverlap,
                maps,error);
        }

        static bool BuildCompressedIndependentMultiLines(
            const IndependentMultiPointSolution &center,
            const IndependentMultiPointSolution &plusH,
            const IndependentMultiPointSolution &minusH,
            const IndependentMultiPointSolution &plusHH,
            const IndependentMultiPointSolution &minusHH,
            const IndependentMultiTrackingMaps &mapPlusH,
            const IndependentMultiTrackingMaps &mapMinusH,
            const IndependentMultiTrackingMaps &mapPlusHH,
            const IndependentMultiTrackingMaps &mapMinusHH,
            const HybridNuclearResonanceRequest &hybrid,
            const HybridNuclearResonanceFieldResponseRequest &fieldResponse,
            const SpectrumRequest &request,
            ResonanceLineSet &lineSet,
            HybridNuclearResonanceReport &report,
            std::string &error)
        {
            if (hybrid.minimumCumulativeOverlapWeight > 0.0)
            {
                error =
                    "compressed hybrid composition is incompatible with path-wise cumulative-overlap pruning; set hybridminimumcumulativeoverlapweight=0";
                return false;
            }

            report = HybridNuclearResonanceReport{};
            report.nucleusCount=center.factors.size();
            report.productNuclearDimension=
                center.productNuclearDimension;
            report.largestDiagonalizedNuclearDimension=
                center.largestDiagonalizedNuclearDimension;
            report.compositionBackend=
                HybridNuclearCompositionMode::Compressed;
            report.maximumComponentsAfterStage.assign(
                center.factors.size(),0);

            report.formalCartesianComponents=1;
            for (const auto &factor:center.factors)
            {
                const std::size_t dimension=
                    static_cast<std::size_t>(factor.dimension);
                if (dimension!=0 &&
                    dimension>
                        std::numeric_limits<std::size_t>::max()/dimension)
                {
                    report.formalCartesianComponents=
                        std::numeric_limits<std::size_t>::max();
                    report.formalCartesianOverflow=true;
                    break;
                }
                const std::size_t local=dimension*dimension;
                if (report.formalCartesianComponents!=0 &&
                    local>
                        std::numeric_limits<std::size_t>::max()/
                            report.formalCartesianComponents)
                {
                    report.formalCartesianComponents=
                        std::numeric_limits<std::size_t>::max();
                    report.formalCartesianOverflow=true;
                    break;
                }
                report.formalCartesianComponents*=local;
            }

            const double productDimension=
                static_cast<double>(center.productNuclearDimension);
            const double h=fieldResponse.fieldStepT;
            const double hh=0.5*h;
            const double omegaMw=2.0*arma::datum::pi*
                request.microwaveFrequencyGHz;
            const arma::uword coreDimension=
                center.core.coreEnergies.n_elem;

            for (arma::uword lower=0;lower<coreDimension;++lower)
            {
                for (arma::uword upper=lower+1;
                     upper<coreDimension;++upper)
                {
                    ++report.coreTransitions;
                    const double corePopulationDifference=
                        center.core.corePopulations(lower)-
                        center.core.corePopulations(upper);
                    const double populationDifference=
                        corePopulationDifference/productDimension;
                    if (std::abs(populationDifference)<
                        request.populationThreshold)
                        continue;

                    TransitionMoment coreMoment;
                    if (center.core.detectionChannelsEigen.empty())
                    {
                        coreMoment=
                            ResonanceTransitionMoments::Evaluate(
                                center.core.muXEigen,
                                center.core.muYEigen,
                                lower,upper);
                    }
                    else if (!ResonanceTransitionMoments::EvaluateResolved(
                            center.core.muXEigen,
                            center.core.muYEigen,
                            center.core.detectionChannelsEigen,
                            lower,upper,coreMoment,error))
                        return false;

                    double corePlusH=0.0,coreMinusH=0.0;
                    double corePlusHH=0.0,coreMinusHH=0.0;
                    if (!TrackedIndependentCoreGap(
                            lower,upper,plusH,mapPlusH,corePlusH) ||
                        !TrackedIndependentCoreGap(
                            lower,upper,minusH,mapMinusH,coreMinusH) ||
                        !TrackedIndependentCoreGap(
                            lower,upper,plusHH,mapPlusHH,corePlusHH) ||
                        !TrackedIndependentCoreGap(
                            lower,upper,minusHH,mapMinusHH,coreMinusHH))
                    {
                        error=
                            "non-finite tracked exact-core transition frequency";
                        return false;
                    }

                    HybridNuclearShiftComponent coreComponent;
                    coreComponent.omegaRadNs=
                        center.core.coreEnergies(upper)-
                        center.core.coreEnergies(lower);
                    coreComponent.slopeH_RadNsT=
                        (corePlusH-coreMinusH)/(2.0*h);
                    coreComponent.slopeHH_RadNsT=
                        (corePlusHH-coreMinusHH)/(2.0*hh);
                    coreComponent.weight=1.0;

                    std::vector<std::vector<
                        HybridNuclearShiftComponent>> factors;
                    factors.reserve(center.factors.size());

                    for (std::size_t k=0;
                         k<center.factors.size();++k)
                    {
                        const auto &factor=center.factors[k];
                        const auto &lowerManifold=
                            factor.nuclear[lower];
                        const auto &upperManifold=
                            factor.nuclear[upper];
                        std::vector<HybridNuclearShiftComponent> local;
                        const std::size_t localCapacity=
                            static_cast<std::size_t>(factor.dimension)*
                            static_cast<std::size_t>(factor.dimension);
                        local.reserve(localCapacity);

                        for (arma::uword r=0;r<factor.dimension;++r)
                        {
                            for (arma::uword s=0;s<factor.dimension;++s)
                            {
                                const arma::cx_double overlap=arma::cdot(
                                    upperManifold.eigenvectors.col(s),
                                    lowerManifold.eigenvectors.col(r));
                                double overlapWeight=std::norm(overlap);
                                if (!std::isfinite(overlapWeight) ||
                                    overlapWeight<0.0 ||
                                    overlapWeight>1.0+1.0e-10)
                                {
                                    error=
                                        "invalid independent nuclear-state overlap weight";
                                    return false;
                                }
                                overlapWeight=std::min(1.0,overlapWeight);
                                if (overlapWeight<factor.overlapThreshold)
                                {
                                    report.pruningApplied=true;
                                    continue;
                                }

                                double shiftPlusH=0.0,shiftMinusH=0.0;
                                double shiftPlusHH=0.0,shiftMinusHH=0.0;
                                if (!TrackedIndependentNuclearShift(
                                        k,lower,upper,r,s,
                                        plusH,mapPlusH,shiftPlusH) ||
                                    !TrackedIndependentNuclearShift(
                                        k,lower,upper,r,s,
                                        minusH,mapMinusH,shiftMinusH) ||
                                    !TrackedIndependentNuclearShift(
                                        k,lower,upper,r,s,
                                        plusHH,mapPlusHH,shiftPlusHH) ||
                                    !TrackedIndependentNuclearShift(
                                        k,lower,upper,r,s,
                                        minusHH,mapMinusHH,shiftMinusHH))
                                {
                                    error=
                                        "non-finite tracked independent nuclear shift";
                                    return false;
                                }

                                HybridNuclearShiftComponent component;
                                component.omegaRadNs=
                                    upperManifold.energies(s)-
                                    lowerManifold.energies(r);
                                component.slopeH_RadNsT=
                                    (shiftPlusH-shiftMinusH)/(2.0*h);
                                component.slopeHH_RadNsT=
                                    (shiftPlusHH-shiftMinusHH)/(2.0*hh);
                                component.weight=overlapWeight;
                                local.push_back(component);
                            }
                        }
                        if (local.empty())
                        {
                            error=
                                "all independent nuclear branches were removed by the overlap threshold";
                            return false;
                        }
                        factors.push_back(std::move(local));
                    }

                    HybridNuclearShiftKernelOptions options;
                    options.mode=HybridNuclearCompositionMode::Compressed;
                    options.fieldTolerance_mT=
                        hybrid.compressionTolerance_mT;
                    options.linewidth_mT=request.linewidth_mT;
                    options.microwaveOmegaRadNs=omegaMw;
                    options.minimumSlopeRadNsT=request.minimumSlope;
                    options.referenceSlopeRadNsT=
                        omegaMw/fieldResponse.fieldT;
                    options.maximumComponents=
                        hybrid.maximumComponentsPerCoreTransition==0
                        ? 65536
                        : hybrid.maximumComponentsPerCoreTransition;

                    std::vector<HybridNuclearShiftComponent> composed;
                    HybridNuclearShiftKernelReport kernelReport;
                    if (!HybridNuclearShiftKernel::Compose(
                            coreComponent,factors,options,
                            composed,kernelReport,error))
                        return false;

                    if (report.generatedComponents >
                        std::numeric_limits<std::size_t>::max()-
                            kernelReport.generatedCandidates)
                        report.generatedComponents=
                            std::numeric_limits<std::size_t>::max();
                    else
                        report.generatedComponents +=
                            kernelReport.generatedCandidates;
                    report.retainedComponents += composed.size();
                    report.mergedComponents +=
                        kernelReport.mergedComponents;
                    report.mergingApplied =
                        report.mergingApplied ||
                        kernelReport.mergedComponents>0;
                    report.maximumIntermediateComponents=std::max(
                        report.maximumIntermediateComponents,
                        kernelReport.maximumIntermediateComponents);
                    report.maximumConvolutionWeightError=std::max(
                        report.maximumConvolutionWeightError,
                        kernelReport.relativeWeightError);
                    for (std::size_t k=0;
                         k<kernelReport.componentsAfterStage.size();++k)
                    {
                        report.maximumComponentsAfterStage[k]=std::max(
                            report.maximumComponentsAfterStage[k],
                            kernelReport.componentsAfterStage[k]);
                    }

                    double retainedWeight=0.0;
                    for (const auto &component:composed)
                        retainedWeight += component.weight;
                    double retainedFraction=
                        retainedWeight/productDimension;
                    if (!std::isfinite(retainedFraction) ||
                        retainedFraction>1.0+1.0e-9)
                    {
                        error=
                            "compressed independent nuclear overlap weight is not normalized";
                        return false;
                    }
                    retainedFraction=std::max(0.0,
                        std::min(1.0,retainedFraction));
                    report.maximumDiscardedNuclearWeightFraction=std::max(
                        report.maximumDiscardedNuclearWeightFraction,
                        1.0-retainedFraction);

                    for (const auto &component:composed)
                    {
                        const double convergenceError=std::abs(
                            component.slopeHH_RadNsT-
                            component.slopeH_RadNsT);
                        const double convergenceScale=std::max(
                            std::abs(component.slopeHH_RadNsT),
                            std::abs(component.slopeH_RadNsT));
                        const double convergenceTolerance=
                            fieldResponse.jacobianAbsoluteTolerance+
                            fieldResponse.jacobianRelativeTolerance*
                                convergenceScale;
                        if (convergenceError>convergenceTolerance)
                        {
                            error=
                                "compressed multi-nucleus finite-difference field derivative did not converge";
                            return false;
                        }

                        const double slope=
                            std::abs(component.slopeHH_RadNsT);
                        if (slope<request.minimumSlope)
                            continue;
                        const double dBdOmega=1.0/slope;
                        if (!std::isfinite(dBdOmega) ||
                            dBdOmega>request.maximumDBdOmega)
                            continue;

                        ResonanceLine line;
                        line.lower=lower;
                        line.upper=upper;
                        line.omega=component.omegaRadNs;
                        line.populationDifference=populationDifference;
                        line.dOmegaDB=slope;
                        line.dBdOmega=dBdOmega;
                        line.moment=ResonanceTransitionMoments::Scale(
                            coreMoment,component.weight);
                        if (!std::isfinite(line.omega) ||
                            !std::isfinite(line.populationDifference) ||
                            !ResonanceTransitionMoments::IsFinite(line.moment))
                        {
                            error=
                                "non-finite compressed multinuclear resonance line";
                            return false;
                        }
                        lineSet.lines.push_back(std::move(line));
                    }
                }
            }

            lineSet.fieldJacobianQualified=true;
            report.outputComponents=lineSet.lines.size();
            return true;
        }

    private:
        static bool GenerateMultiNucleusFirstOrder(
            const HybridNuclearResonancePoint &point,
            const SpectrumRequest &request,
            ResonanceLineSet &lineSet,
            HybridNuclearResonanceReport &report,
            std::string &error)
        {
            error.clear();
            lineSet.lines.clear();
            lineSet.fieldJacobianQualified = false;
            report =
                HybridNuclearResonanceReport{};

            IndependentMultiPointSolution solution;
            if (!SolveIndependentMultiPoint(
                    point,request,
                    solution,error))
                return false;

            std::vector<IndependentMultiComponent>
                components;
            if (!BuildIndependentMultiComponents(
                    solution,point.hybrid,request,
                    components,report,error))
                return false;

            lineSet.lines.reserve(
                components.size());
            for (const auto &component:components)
                lineSet.lines.push_back(
                    component.line);

            report.outputComponents =
                lineSet.lines.size();
            return true;
        }

        static bool GenerateMultiNucleusFirstOrderFiniteDifference(
            const HybridNuclearResonancePointProvider &provider,
            const HybridNuclearResonanceFieldResponseRequest &fieldResponse,
            const SpectrumRequest &request,
            ResonanceLineSet &lineSet,
            HybridNuclearResonanceReport &report,
            std::string &error)
        {
            error.clear();
            lineSet.lines.clear();
            lineSet.fieldJacobianQualified = false;
            report =
                HybridNuclearResonanceReport{};

            if (!provider)
            {
                error =
                    "multi-nucleus field-response point provider is not set";
                return false;
            }

            if (!std::isfinite(fieldResponse.fieldT) ||
                !std::isfinite(
                    fieldResponse.fieldStepT) ||
                fieldResponse.fieldT <= 0.0 ||
                fieldResponse.fieldStepT <= 0.0 ||
                fieldResponse.fieldStepT >=
                    fieldResponse.fieldT ||
                !std::isfinite(
                    fieldResponse.
                        minimumCoreStateOverlap) ||
                !std::isfinite(
                    fieldResponse.
                        minimumNuclearStateOverlap) ||
                fieldResponse.
                    minimumCoreStateOverlap <= 0.0 ||
                fieldResponse.
                    minimumCoreStateOverlap > 1.0 ||
                fieldResponse.
                    minimumNuclearStateOverlap <= 0.0 ||
                fieldResponse.
                    minimumNuclearStateOverlap > 1.0 ||
                !std::isfinite(
                    fieldResponse.
                        jacobianRelativeTolerance) ||
                !std::isfinite(
                    fieldResponse.
                        jacobianAbsoluteTolerance) ||
                fieldResponse.
                    jacobianRelativeTolerance < 0.0 ||
                fieldResponse.
                    jacobianAbsoluteTolerance < 0.0)
            {
                error =
                    "invalid multi-nucleus field-response request";
                return false;
            }

            HybridNuclearResonancePoint
                centerPoint;
            if (!provider(
                    fieldResponse.fieldT,
                    centerPoint,error))
            {
                if (error.empty())
                    error =
                        "multi-nucleus field-response center provider failed";
                return false;
            }

            IndependentMultiPointSolution center;
            if (!SolveIndependentMultiPoint(
                    centerPoint,request,
                    center,error))
                return false;

            std::size_t formalComponents=1;
            bool formalOverflow=false;
            for (const auto &factor:center.factors)
            {
                const std::size_t dimension=
                    static_cast<std::size_t>(factor.dimension);
                std::size_t localComponents=0;
                if (dimension != 0 &&
                    dimension >
                        std::numeric_limits<std::size_t>::max()/dimension)
                {
                    localComponents=
                        std::numeric_limits<std::size_t>::max();
                    formalOverflow=true;
                }
                else
                    localComponents=dimension*dimension;
                if (formalComponents != 0 &&
                    localComponents >
                        std::numeric_limits<std::size_t>::max()/
                            formalComponents)
                {
                    formalComponents=
                        std::numeric_limits<std::size_t>::max();
                    formalOverflow=true;
                }
                else
                    formalComponents*=localComponents;
            }

            HybridNuclearCompositionMode backend=
                centerPoint.hybrid.compositionMode;
            if (backend==HybridNuclearCompositionMode::Auto)
            {
                backend=
                    formalOverflow ||
                    formalComponents>
                        (centerPoint.hybrid.
                            maximumComponentsPerCoreTransition==0
                         ? 65536
                         : centerPoint.hybrid.
                            maximumComponentsPerCoreTransition)
                    ? HybridNuclearCompositionMode::Compressed
                    : HybridNuclearCompositionMode::Explicit;
            }
            if (backend==HybridNuclearCompositionMode::Compressed &&
                centerPoint.hybrid.compressionTolerance_mT<=0.0)
            {
                error=
                    "automatic compressed hybrid composition requires a positive compression tolerance";
                return false;
            }
            if (backend==HybridNuclearCompositionMode::Explicit &&
                centerPoint.hybrid.mergeFrequencyToleranceRadNs>0.0)
            {
                error=
                    "multi-nucleus field response requires unmerged center components";
                return false;
            }
            if (backend==HybridNuclearCompositionMode::Explicit &&
                (formalOverflow ||
                 formalComponents>
                    (centerPoint.hybrid.
                        maximumComponentsPerCoreTransition==0
                     ? 65536
                     : centerPoint.hybrid.
                        maximumComponentsPerCoreTransition)))
            {
                error=
                    "explicit hybrid nuclear component cap exceeded before allocation";
                return false;
            }

            const double h =
                fieldResponse.fieldStepT;
            const double hh =
                0.5*h;

            IndependentMultiPointSolution
                plusH,minusH,plusHH,minusHH;
            IndependentMultiTrackingMaps
                mapPlusH,mapMinusH,
                mapPlusHH,mapMinusHH;

            if (!SolveTrackedIndependentMultiDisplacement(
                    provider,
                    fieldResponse.fieldT+h,
                    request,center,
                    fieldResponse.
                        minimumCoreStateOverlap,
                    fieldResponse.
                        minimumNuclearStateOverlap,
                    plusH,mapPlusH,error) ||
                !SolveTrackedIndependentMultiDisplacement(
                    provider,
                    fieldResponse.fieldT-h,
                    request,center,
                    fieldResponse.
                        minimumCoreStateOverlap,
                    fieldResponse.
                        minimumNuclearStateOverlap,
                    minusH,mapMinusH,error) ||
                !SolveTrackedIndependentMultiDisplacement(
                    provider,
                    fieldResponse.fieldT+hh,
                    request,center,
                    fieldResponse.
                        minimumCoreStateOverlap,
                    fieldResponse.
                        minimumNuclearStateOverlap,
                    plusHH,mapPlusHH,error) ||
                !SolveTrackedIndependentMultiDisplacement(
                    provider,
                    fieldResponse.fieldT-hh,
                    request,center,
                    fieldResponse.
                        minimumCoreStateOverlap,
                    fieldResponse.
                        minimumNuclearStateOverlap,
                    minusHH,mapMinusHH,error))
                return false;

            if (backend==HybridNuclearCompositionMode::Compressed)
            {
                return BuildCompressedIndependentMultiLines(
                    center,plusH,minusH,plusHH,minusHH,
                    mapPlusH,mapMinusH,mapPlusHH,mapMinusHH,
                    centerPoint.hybrid,fieldResponse,request,
                    lineSet,report,error);
            }

            std::vector<IndependentMultiComponent>
                components;
            HybridNuclearResonanceRequest explicitRequest=
                centerPoint.hybrid;
            explicitRequest.compositionMode=
                HybridNuclearCompositionMode::Explicit;
            if (!BuildIndependentMultiComponents(
                    center,explicitRequest,
                    request,components,
                    report,error))
                return false;
            report.formalCartesianComponents=formalComponents;
            report.formalCartesianOverflow=formalOverflow;
            report.compositionBackend=
                HybridNuclearCompositionMode::Explicit;

            for (auto component:components)
            {
                double omegaPlusH=0.0;
                double omegaMinusH=0.0;
                double omegaPlusHH=0.0;
                double omegaMinusHH=0.0;

                if (!TrackedIndependentMultiFrequency(
                        component,plusH,
                        mapPlusH,omegaPlusH) ||
                    !TrackedIndependentMultiFrequency(
                        component,minusH,
                        mapMinusH,omegaMinusH) ||
                    !TrackedIndependentMultiFrequency(
                        component,plusHH,
                        mapPlusHH,omegaPlusHH) ||
                    !TrackedIndependentMultiFrequency(
                        component,minusHH,
                        mapMinusHH,omegaMinusHH))
                {
                    error =
                        "non-finite tracked multi-nucleus transition frequency";
                    return false;
                }

                const double slopeH =
                    (omegaPlusH-omegaMinusH)/
                    (2.0*h);
                const double slopeHH =
                    (omegaPlusHH-omegaMinusHH)/
                    (2.0*hh);

                if (!std::isfinite(slopeH) ||
                    !std::isfinite(slopeHH))
                {
                    error =
                        "non-finite multi-nucleus finite-difference field derivative";
                    return false;
                }

                const double convergenceError =
                    std::abs(
                        slopeHH-slopeH);
                const double convergenceScale =
                    std::max(
                        std::abs(slopeHH),
                        std::abs(slopeH));
                const double convergenceTolerance =
                    fieldResponse.
                        jacobianAbsoluteTolerance +
                    fieldResponse.
                        jacobianRelativeTolerance*
                        convergenceScale;

                if (convergenceError >
                    convergenceTolerance)
                {
                    error =
                        "multi-nucleus finite-difference field derivative did not converge";
                    return false;
                }

                const double slope =
                    std::abs(slopeHH);
                if (slope <
                    request.minimumSlope)
                    continue;

                const double dBdOmega =
                    1.0/slope;
                if (!std::isfinite(dBdOmega) ||
                    dBdOmega >
                        request.maximumDBdOmega)
                    continue;

                component.line.dOmegaDB =
                    slope;
                component.line.dBdOmega =
                    dBdOmega;
                lineSet.lines.push_back(
                    component.line);
            }

            lineSet.fieldJacobianQualified =
                true;
            report.outputComponents =
                lineSet.lines.size();
            return true;
        }

        static bool GenerateSingleNucleusFirstOrder(
            const HybridNuclearResonancePoint &point,
            const SpectrumRequest &request,
            ResonanceLineSet &lineSet,
            std::string &error)
        {
            error.clear();
            lineSet.lines.clear();
            lineSet.fieldJacobianQualified =
                point.hybrid.nuclei.front().
                    fieldIndependentProjection;

            PointSolution solution;
            if (!SolvePoint(point,request,solution,error))
                return false;

            std::vector<HybridComponent> components;
            if (!BuildComponents(
                    solution,request,true,components,error))
                return false;

            for (const auto &component : components)
                lineSet.lines.push_back(component.line);

            return true;
        }

        static bool GenerateSingleNucleusFirstOrderFiniteDifference(
            const HybridNuclearResonancePointProvider &provider,
            const HybridNuclearResonanceFieldResponseRequest &fieldResponse,
            const SpectrumRequest &request,
            ResonanceLineSet &lineSet,
            std::string &error)
        {
            error.clear();
            lineSet.lines.clear();
            lineSet.fieldJacobianQualified = false;

            if (!provider)
            {
                error =
                    "hybrid field-response point provider is not set";
                return false;
            }
            if (!std::isfinite(fieldResponse.fieldT) ||
                !std::isfinite(fieldResponse.fieldStepT) ||
                fieldResponse.fieldT <= 0.0 ||
                fieldResponse.fieldStepT <= 0.0 ||
                fieldResponse.fieldStepT >= fieldResponse.fieldT ||
                !std::isfinite(
                    fieldResponse.minimumCoreStateOverlap) ||
                !std::isfinite(
                    fieldResponse.minimumNuclearStateOverlap) ||
                fieldResponse.minimumCoreStateOverlap <= 0.0 ||
                fieldResponse.minimumCoreStateOverlap > 1.0 ||
                fieldResponse.minimumNuclearStateOverlap <= 0.0 ||
                fieldResponse.minimumNuclearStateOverlap > 1.0 ||
                !std::isfinite(
                    fieldResponse.jacobianRelativeTolerance) ||
                !std::isfinite(
                    fieldResponse.jacobianAbsoluteTolerance) ||
                fieldResponse.jacobianRelativeTolerance < 0.0 ||
                fieldResponse.jacobianAbsoluteTolerance < 0.0)
            {
                error =
                    "invalid hybrid field-response request";
                return false;
            }

            HybridNuclearResonancePoint centerPoint;
            if (!provider(
                    fieldResponse.fieldT,centerPoint,error))
            {
                if (error.empty())
                    error =
                        "hybrid field-response center provider failed";
                return false;
            }

            PointSolution center;
            if (!SolvePoint(
                    centerPoint,request,center,error))
                return false;

            std::vector<HybridComponent> components;
            if (!BuildComponents(
                    center,request,false,components,error))
                return false;

            const double h = fieldResponse.fieldStepT;
            const double hh = 0.5*h;

            PointSolution plusH,minusH,plusHH,minusHH;
            std::vector<arma::uword>
                corePlusH,coreMinusH,corePlusHH,coreMinusHH;
            std::vector<std::vector<arma::uword>>
                nucPlusH,nucMinusH,nucPlusHH,nucMinusHH;

            if (!SolveTrackedDisplacement(
                    provider,fieldResponse.fieldT+h,
                    request,center,
                    fieldResponse.minimumCoreStateOverlap,
                    fieldResponse.minimumNuclearStateOverlap,
                    plusH,corePlusH,nucPlusH,error) ||
                !SolveTrackedDisplacement(
                    provider,fieldResponse.fieldT-h,
                    request,center,
                    fieldResponse.minimumCoreStateOverlap,
                    fieldResponse.minimumNuclearStateOverlap,
                    minusH,coreMinusH,nucMinusH,error) ||
                !SolveTrackedDisplacement(
                    provider,fieldResponse.fieldT+hh,
                    request,center,
                    fieldResponse.minimumCoreStateOverlap,
                    fieldResponse.minimumNuclearStateOverlap,
                    plusHH,corePlusHH,nucPlusHH,error) ||
                !SolveTrackedDisplacement(
                    provider,fieldResponse.fieldT-hh,
                    request,center,
                    fieldResponse.minimumCoreStateOverlap,
                    fieldResponse.minimumNuclearStateOverlap,
                    minusHH,coreMinusHH,nucMinusHH,error))
                return false;

            for (auto component : components)
            {
                double omegaPlusH=0.0,omegaMinusH=0.0;
                double omegaPlusHH=0.0,omegaMinusHH=0.0;
                if (!TrackedFrequency(
                        component,plusH,
                        corePlusH,nucPlusH,omegaPlusH) ||
                    !TrackedFrequency(
                        component,minusH,
                        coreMinusH,nucMinusH,omegaMinusH) ||
                    !TrackedFrequency(
                        component,plusHH,
                        corePlusHH,nucPlusHH,omegaPlusHH) ||
                    !TrackedFrequency(
                        component,minusHH,
                        coreMinusHH,nucMinusHH,omegaMinusHH))
                {
                    error =
                        "non-finite tracked hybrid transition frequency";
                    return false;
                }

                const double slopeH =
                    (omegaPlusH-omegaMinusH)/(2.0*h);
                const double slopeHH =
                    (omegaPlusHH-omegaMinusHH)/(2.0*hh);

                if (!std::isfinite(slopeH) ||
                    !std::isfinite(slopeHH))
                {
                    error =
                        "non-finite hybrid finite-difference field derivative";
                    return false;
                }

                const double convergenceError =
                    std::abs(slopeHH-slopeH);
                const double convergenceScale =
                    std::max(std::abs(slopeHH),
                             std::abs(slopeH));
                const double convergenceTolerance =
                    fieldResponse.jacobianAbsoluteTolerance +
                    fieldResponse.jacobianRelativeTolerance*
                    convergenceScale;

                if (convergenceError > convergenceTolerance)
                {
                    error =
                        "hybrid finite-difference field derivative did not converge";
                    return false;
                }

                const double slope = std::abs(slopeHH);
                if (slope < request.minimumSlope)
                    continue;

                const double dBdOmega = 1.0/slope;
                if (!std::isfinite(dBdOmega) ||
                    dBdOmega > request.maximumDBdOmega)
                    continue;

                component.line.dOmegaDB = slope;
                component.line.dBdOmega = dBdOmega;
                lineSet.lines.push_back(component.line);
            }

            lineSet.fieldJacobianQualified = true;
            return true;
        }

    public:
        static bool GenerateFirstOrder(
            const HybridNuclearResonancePoint &point,
            const SpectrumRequest &request,
            ResonanceLineSet &lineSet,
            HybridNuclearResonanceReport &report,
            std::string &error)
        {
            report = HybridNuclearResonanceReport{};

            if (point.hybrid.nuclei.empty())
            {
                error =
                    "hybrid resonance point requires at least one perturbative nucleus";
                lineSet.lines.clear();
                lineSet.fieldJacobianQualified=false;
                return false;
            }

            if (point.hybrid.nuclei.size()==1 &&
                point.hybrid.nuclei.front().
                    fieldIndependentProjection)
            {
                const auto &nucleus =
                    point.hybrid.nuclei.front();

                const bool ok =
                    GenerateSingleNucleusFirstOrder(
                        point,
                        request,lineSet,error);
                if (!ok)
                    return false;

                report.nucleusCount=1;
                report.productNuclearDimension=
                    static_cast<std::size_t>(
                        nucleus.nuclearDimension);
                report.largestDiagonalizedNuclearDimension=
                    static_cast<std::size_t>(
                        nucleus.nuclearDimension);
                report.outputComponents=
                    lineSet.lines.size();
                return true;
            }

            return GenerateMultiNucleusFirstOrder(
                point,
                request,lineSet,report,error);
        }

        static bool GenerateFirstOrderFiniteDifference(
            const HybridNuclearResonancePointProvider &provider,
            const HybridNuclearResonanceFieldResponseRequest &fieldResponse,
            const SpectrumRequest &request,
            ResonanceLineSet &lineSet,
            HybridNuclearResonanceReport &report,
            std::string &error)
        {
            return GenerateMultiNucleusFirstOrderFiniteDifference(
                provider,fieldResponse,request,
                lineSet,report,error);
        }
    };
}

#endif
