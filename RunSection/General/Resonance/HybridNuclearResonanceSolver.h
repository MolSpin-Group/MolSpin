/////////////////////////////////////////////////////////////////////////
// HybridNuclearResonanceSolver (RunSection::General::Resonance)
// ------------------
// Electron/core-exact, one-nucleus perturbative resonance backend.
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
#include "ResonanceTransitionMoments.h"
#include "ResonanceTypes.h"

#include <armadillo>
#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

namespace RunSection::General::Resonance
{
    struct OneNucleusHybridRequest
    {
        arma::cx_mat hyperfineCoreNuclear;
        arma::cx_mat nuclearHamiltonian;
        arma::sp_cx_mat nuclearDHdB;
        arma::uword nuclearDimension = 0;
        double overlapThreshold = 1.0e-14;
        bool fieldIndependentProjection = false;
    };

    struct OneNucleusHybridPoint
    {
        arma::sp_cx_mat coreHamiltonian;
        arma::cx_mat coreDensity;
        arma::sp_cx_mat coreDHdB;
        arma::cx_mat coreMuX;
        arma::cx_mat coreMuY;
        OneNucleusHybridRequest hybrid;
    };

    using OneNucleusHybridPointProvider =
        std::function<bool(double, OneNucleusHybridPoint &, std::string &)>;

    struct OneNucleusHybridFieldResponseRequest
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

        static bool ValidatePoint(const OneNucleusHybridPoint &point,
            const SpectrumRequest &request, std::string &error)
        {
            const arma::uword coreDimension =
                point.coreHamiltonian.n_rows;
            const arma::uword nuclearDimension =
                point.hybrid.nuclearDimension;

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
            if (nuclearDimension < 2 ||
                point.hybrid.nuclearHamiltonian.n_rows != nuclearDimension ||
                point.hybrid.nuclearHamiltonian.n_cols != nuclearDimension ||
                point.hybrid.nuclearDHdB.n_rows != nuclearDimension ||
                point.hybrid.nuclearDHdB.n_cols != nuclearDimension ||
                point.hybrid.hyperfineCoreNuclear.n_rows !=
                    coreDimension*nuclearDimension ||
                point.hybrid.hyperfineCoreNuclear.n_cols !=
                    coreDimension*nuclearDimension)
            {
                error = "invalid one-nucleus hybrid dimensions";
                return false;
            }
            if (!point.coreDensity.is_finite() ||
                !point.hybrid.nuclearHamiltonian.is_finite() ||
                !point.hybrid.hyperfineCoreNuclear.is_finite() ||
                !std::isfinite(point.hybrid.overlapThreshold) ||
                point.hybrid.overlapThreshold < 0.0 ||
                point.hybrid.overlapThreshold > 1.0 ||
                request.populationThreshold < 0.0 ||
                request.minimumSlope < 0.0 ||
                request.maximumDBdOmega <= 0.0)
            {
                error = "invalid one-nucleus hybrid numerical input";
                return false;
            }
            return true;
        }

        static bool SolvePoint(const OneNucleusHybridPoint &point,
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
                    solution.muXEigen,solution.muYEigen,error))
                return false;

            solution.nuclearDimension =
                point.hybrid.nuclearDimension;
            solution.overlapThreshold =
                point.hybrid.overlapThreshold;
            const arma::uword coreDimension =
                solution.coreEnergies.n_elem;
            solution.nuclear.resize(coreDimension);

            for (arma::uword a=0; a<coreDimension; ++a)
            {
                arma::cx_mat projected;
                if (!PartialCoreExpectation(
                        point.hybrid.hyperfineCoreNuclear,
                        solution.coreEigenvectors.col(a),
                        solution.nuclearDimension,projected))
                {
                    error =
                        "failed to project the hyperfine operator onto a core state";
                    return false;
                }

                arma::cx_mat effectiveNuclear =
                    point.hybrid.nuclearHamiltonian + projected;
                effectiveNuclear =
                    0.5*(effectiveNuclear + effectiveNuclear.t());

                if (!arma::eig_sym(
                        solution.nuclear[a].energies,
                        solution.nuclear[a].eigenvectors,
                        effectiveNuclear))
                {
                    error =
                        "failed to diagonalize an effective nuclear Hamiltonian";
                    return false;
                }

                if (!ResonanceFieldJacobian::ResolveDegenerateSubspaces(
                        solution.nuclear[a].energies,
                        solution.nuclear[a].eigenvectors,
                        point.hybrid.nuclearDHdB,
                        solution.nuclear[a].dEdB,error))
                    return false;
            }

            return true;
        }

        static bool BuildComponents(const PointSolution &solution,
            const SpectrumRequest &request, bool applyAnalyticSlope,
            std::vector<HybridComponent> &components)
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
                    const TransitionMoment coreMoment =
                        ResonanceTransitionMoments::Evaluate(
                            solution.muXEigen,solution.muYEigen,
                            lower,upper);

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
                            component.line.moment.x =
                                coreMoment.x*overlapWeight;
                            component.line.moment.y =
                                coreMoment.y*overlapWeight;
                            component.line.moment.perpendicular =
                                coreMoment.perpendicular*overlapWeight;

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
            const OneNucleusHybridPointProvider &provider,
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
            OneNucleusHybridPoint point;
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

    public:
        static bool GenerateFirstOrder(
            const arma::sp_cx_mat &coreHamiltonian,
            const arma::cx_mat &coreDensity,
            const arma::sp_cx_mat &coreDHdB,
            const arma::cx_mat &coreMuX,
            const arma::cx_mat &coreMuY,
            const OneNucleusHybridRequest &hybrid,
            const SpectrumRequest &request,
            ResonanceLineSet &lineSet,
            std::string &error)
        {
            OneNucleusHybridPoint point;
            point.coreHamiltonian = coreHamiltonian;
            point.coreDensity = coreDensity;
            point.coreDHdB = coreDHdB;
            point.coreMuX = coreMuX;
            point.coreMuY = coreMuY;
            point.hybrid = hybrid;

            error.clear();
            lineSet.lines.clear();
            lineSet.fieldJacobianQualified =
                hybrid.fieldIndependentProjection;

            PointSolution solution;
            if (!SolvePoint(point,request,solution,error))
                return false;

            std::vector<HybridComponent> components;
            if (!BuildComponents(
                    solution,request,true,components))
                return false;

            for (const auto &component : components)
                lineSet.lines.push_back(component.line);

            return true;
        }

        static bool GenerateFirstOrderFiniteDifference(
            const OneNucleusHybridPointProvider &provider,
            const OneNucleusHybridFieldResponseRequest &fieldResponse,
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

            OneNucleusHybridPoint centerPoint;
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
                    center,request,false,components))
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
    };
}

#endif
