/////////////////////////////////////////////////////////////////////////
// ExactResonanceSolver (RunSection::General::Resonance)
// ------------------
// Exact full-Hilbert resonance-line backend. This class owns only the mapping
// from one complete eigensystem to backend-neutral ResonanceLine objects.
// Powder sampling, line broadening, output formatting and task ownership remain
// outside this solver.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_Resonance_ExactResonanceSolver
#define MOD_RunSection_General_Resonance_ExactResonanceSolver

#include "ResonanceFieldJacobian.h"
#include "ResonanceTransitionDetector.h"
#include "ResonanceTransitionMoments.h"
#include "ResonanceTypes.h"

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace RunSection::General::Resonance
{
    class ExactResonanceSolver
    {
    private:
        static bool ValidateOperators(
            arma::uword dim,
            const arma::cx_mat &density,
            const arma::sp_cx_mat &dHdB,
            const arma::cx_mat &muX,
            const arma::cx_mat &muY,
            const std::vector<ResonanceDetectionOperator> &detectionChannels,
            std::string &error)
        {
            if (density.n_rows != dim || density.n_cols != dim)
            {
                error =
                    "density matrix dimension does not match the resonance Hamiltonian";
                return false;
            }
            if (dHdB.n_rows != dim || dHdB.n_cols != dim ||
                muX.n_rows != dim || muX.n_cols != dim ||
                muY.n_rows != dim || muY.n_cols != dim)
            {
                error =
                    "resonance operator dimension does not match the Hamiltonian";
                return false;
            }
            if (!density.is_finite() ||
                !arma::cx_mat(dHdB).is_finite() ||
                !muX.is_finite() ||
                !muY.is_finite())
            {
                error =
                    "resonance density or operator contains non-finite values";
                return false;
            }

            if (!detectionChannels.empty())
            {
                arma::cx_mat sumX(
                    dim,dim,arma::fill::zeros);
                arma::cx_mat sumY(
                    dim,dim,arma::fill::zeros);

                for (const auto &channel:detectionChannels)
                {
                    if (channel.x.n_rows!=dim ||
                        channel.x.n_cols!=dim ||
                        channel.y.n_rows!=dim ||
                        channel.y.n_cols!=dim ||
                        !channel.x.is_finite() ||
                        !channel.y.is_finite())
                    {
                        error =
                            "resolved resonance detection channel dimensions do not match the Hamiltonian";
                        return false;
                    }
                    sumX+=channel.x;
                    sumY+=channel.y;
                }

                const double scale=std::max({
                    1.0,
                    arma::norm(muX,"fro"),
                    arma::norm(muY,"fro"),
                    arma::norm(sumX,"fro"),
                    arma::norm(sumY,"fro")
                });

                if (arma::norm(sumX-muX,"fro")>
                        1.0e-12*scale ||
                    arma::norm(sumY-muY,"fro")>
                        1.0e-12*scale)
                {
                    error =
                        "resolved resonance detection channels do not sum to total transverse operators";
                    return false;
                }
            }
            return true;
        }

        static bool GenerateFromValidatedEigensystem(
            const arma::vec &energies,
            arma::cx_mat eigenvectors,
            const arma::cx_mat &density,
            const arma::sp_cx_mat &dHdB,
            const arma::cx_mat &muX,
            const arma::cx_mat &muY,
            const SpectrumRequest &request,
            ResonanceLineSet &lineSet,
            std::string &error,
            const std::vector<ResonanceDetectionOperator> &detectionChannels)
        {
            arma::vec dEdB;
            if (!ResonanceFieldJacobian::ResolveDegenerateSubspaces(
                    energies,eigenvectors,dHdB,dEdB,error))
                return false;

            const arma::cx_mat densityEigen =
                eigenvectors.t()*density*eigenvectors;
            const arma::vec populations =
                arma::real(densityEigen.diag());
            if (!populations.is_finite())
            {
                error =
                    "non-finite eigenbasis populations";
                return false;
            }

            std::vector<Transition> transitions;
            if (!ResonanceTransitionDetector::Detect(
                    energies,populations,dEdB,0.0,
                    transitions,error,
                    request.populationThreshold,
                    request.minimumSlope,
                    request.maximumDBdOmega))
                return false;

            arma::cx_mat muXEigen,muYEigen;
            if (!ResonanceTransitionMoments::Transform(
                    eigenvectors,muX,muY,
                    muXEigen,muYEigen,error))
                return false;

            std::vector<ResonanceDetectionOperator>
                detectionChannelsEigen;
            if (!ResonanceTransitionMoments::TransformChannels(
                    eigenvectors,detectionChannels,
                    detectionChannelsEigen,error))
                return false;

            lineSet.lines.reserve(transitions.size());
            for (const auto &transition:transitions)
            {
                ResonanceLine line;
                line.lower=transition.lower;
                line.upper=transition.upper;
                line.omega=transition.omega;
                line.populationDifference=
                    transition.populationDifference;
                line.dOmegaDB=transition.dOmegaDB;
                line.dBdOmega=transition.dBdOmega;

                if (detectionChannelsEigen.empty())
                {
                    line.moment=
                        ResonanceTransitionMoments::Evaluate(
                            muXEigen,muYEigen,
                            transition.lower,
                            transition.upper);
                }
                else if (!ResonanceTransitionMoments::
                        EvaluateResolved(
                            muXEigen,muYEigen,
                            detectionChannelsEigen,
                            transition.lower,
                            transition.upper,
                            line.moment,error))
                {
                    return false;
                }

                lineSet.lines.push_back(
                    std::move(line));
            }
            return true;
        }

    public:
        static bool Generate(
            const arma::sp_cx_mat &hamiltonian,
            const arma::cx_mat &density,
            const arma::sp_cx_mat &dHdB,
            const arma::cx_mat &muX,
            const arma::cx_mat &muY,
            const SpectrumRequest &request,
            ResonanceLineSet &lineSet,
            std::string &error,
            const std::vector<ResonanceDetectionOperator> &detectionChannels = {})
        {
            error.clear();
            lineSet.lines.clear();
            lineSet.fieldJacobianQualified=true;

            const arma::uword dim=hamiltonian.n_rows;
            if (dim==0 || hamiltonian.n_cols!=dim)
            {
                error =
                    "resonance Hamiltonian must be non-empty and square";
                return false;
            }
            if (!arma::cx_mat(hamiltonian).is_finite())
            {
                error =
                    "resonance Hamiltonian contains non-finite values";
                return false;
            }
            if (!ValidateOperators(
                    dim,density,dHdB,muX,muY,
                    detectionChannels,error))
                return false;

            arma::vec energies;
            arma::cx_mat eigenvectors;
            if (!arma::eig_sym(
                    energies,eigenvectors,
                    arma::cx_mat(hamiltonian)))
            {
                error =
                    "failed to diagonalize the resonance Hamiltonian";
                return false;
            }

            return GenerateFromValidatedEigensystem(
                energies,eigenvectors,density,dHdB,
                muX,muY,request,lineSet,error,
                detectionChannels);
        }

        // Same canonical Generate operation for callers that already possess a
        // complete orthonormal eigensystem, e.g. an Mz-block accelerated task.
        static bool Generate(
            const arma::vec &energies,
            const arma::cx_mat &eigenvectors,
            const arma::cx_mat &density,
            const arma::sp_cx_mat &dHdB,
            const arma::cx_mat &muX,
            const arma::cx_mat &muY,
            const SpectrumRequest &request,
            ResonanceLineSet &lineSet,
            std::string &error,
            const std::vector<ResonanceDetectionOperator> &detectionChannels = {})
        {
            error.clear();
            lineSet.lines.clear();
            lineSet.fieldJacobianQualified=true;

            const arma::uword dim=energies.n_elem;
            if (dim==0 ||
                eigenvectors.n_rows!=dim ||
                eigenvectors.n_cols!=dim ||
                !energies.is_finite() ||
                !eigenvectors.is_finite())
            {
                error =
                    "resonance eigensystem is empty, non-finite, or dimensionally inconsistent";
                return false;
            }
            if (!ValidateOperators(
                    dim,density,dHdB,muX,muY,
                    detectionChannels,error))
                return false;

            const arma::cx_mat gram=
                eigenvectors.t()*eigenvectors;
            const arma::cx_mat identity=
                arma::eye<arma::cx_mat>(dim,dim);
            const double orthogonalityError=
                arma::norm(gram-identity,"fro");
            if (!std::isfinite(orthogonalityError) ||
                orthogonalityError>
                    1.0e-10*std::sqrt(
                        static_cast<double>(dim)))
            {
                error =
                    "resonance eigensystem eigenvectors are not orthonormal";
                return false;
            }

            for (arma::uword i=1;i<dim;++i)
            {
                const double scale=std::max({
                    1.0,
                    std::abs(energies(i-1)),
                    std::abs(energies(i))
                });
                if (energies(i)+1.0e-12*scale<
                    energies(i-1))
                {
                    error =
                        "resonance eigensystem energies must be sorted ascending";
                    return false;
                }
            }

            return GenerateFromValidatedEigensystem(
                energies,eigenvectors,density,dHdB,
                muX,muY,request,lineSet,error,
                detectionChannels);
        }
    };
}

#endif
