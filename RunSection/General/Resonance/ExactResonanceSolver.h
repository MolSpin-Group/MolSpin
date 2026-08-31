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
#include <string>
#include <vector>

namespace RunSection::General::Resonance
{
    class ExactResonanceSolver
    {
    public:
        static bool Generate(const arma::sp_cx_mat &hamiltonian,
            const arma::cx_mat &density, const arma::sp_cx_mat &dHdB,
            const arma::cx_mat &muX, const arma::cx_mat &muY,
            const SpectrumRequest &request, ResonanceLineSet &lineSet,
            std::string &error,
            const std::vector<ResonanceDetectionOperator> &detectionChannels = {})
        {
            error.clear();
            lineSet.lines.clear();
            lineSet.fieldJacobianQualified = true;

            const arma::uword dim = hamiltonian.n_rows;
            if (dim == 0 || hamiltonian.n_cols != dim)
            {
                error = "resonance Hamiltonian must be non-empty and square";
                return false;
            }
            if (density.n_rows != dim || density.n_cols != dim)
            {
                error = "density matrix dimension does not match the resonance Hamiltonian";
                return false;
            }
            if (dHdB.n_rows != dim || dHdB.n_cols != dim ||
                muX.n_rows != dim || muX.n_cols != dim ||
                muY.n_rows != dim || muY.n_cols != dim)
            {
                error = "resonance operator dimension does not match the Hamiltonian";
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

                if (arma::norm(
                        sumX-muX,"fro")>
                        1.0e-12*scale ||
                    arma::norm(
                        sumY-muY,"fro")>
                        1.0e-12*scale)
                {
                    error =
                        "resolved resonance detection channels do not sum to total transverse operators";
                    return false;
                }
            }

            arma::vec energies;
            arma::cx_mat eigenvectors;
            if (!arma::eig_sym(energies, eigenvectors, arma::cx_mat(hamiltonian)))
            {
                error = "failed to diagonalize the resonance Hamiltonian";
                return false;
            }

            arma::vec dEdB;
            if (!ResonanceFieldJacobian::ResolveDegenerateSubspaces(
                    energies, eigenvectors, dHdB, dEdB, error))
                return false;

            const arma::cx_mat densityEigen = eigenvectors.t() * density * eigenvectors;
            const arma::vec populations = arma::real(densityEigen.diag());
            if (!populations.is_finite())
            {
                error = "non-finite eigenbasis populations";
                return false;
            }

            // Transition selection thresholds are frequency-independent.
            // The common spectrum evaluator reconstructs microwave detuning.
            std::vector<Transition> transitions;
            if (!ResonanceTransitionDetector::Detect(energies, populations, dEdB, 0.0,
                    transitions, error, request.populationThreshold, request.minimumSlope,
                    request.maximumDBdOmega))
                return false;

            arma::cx_mat muXEigen, muYEigen;
            if (!ResonanceTransitionMoments::Transform(eigenvectors, muX, muY,
                    muXEigen, muYEigen, error))
                return false;

            std::vector<ResonanceDetectionOperator>
                detectionChannelsEigen;
            if (!ResonanceTransitionMoments::TransformChannels(
                    eigenvectors,detectionChannels,
                    detectionChannelsEigen,error))
                return false;

            lineSet.lines.reserve(transitions.size());
            for (const auto &transition : transitions)
            {
                ResonanceLine line;
                line.lower = transition.lower;
                line.upper = transition.upper;
                line.omega = transition.omega;
                line.populationDifference = transition.populationDifference;
                line.dOmegaDB = transition.dOmegaDB;
                line.dBdOmega = transition.dBdOmega;
                if (detectionChannelsEigen.empty())
                {
                    line.moment =
                        ResonanceTransitionMoments::Evaluate(
                            muXEigen,muYEigen,
                            transition.lower,transition.upper);
                }
                else if (!ResonanceTransitionMoments::EvaluateResolved(
                        muXEigen,muYEigen,
                        detectionChannelsEigen,
                        transition.lower,transition.upper,
                        line.moment,error))
                {
                    return false;
                }
                lineSet.lines.push_back(line);
            }

            return true;
        }
    };
}

#endif
