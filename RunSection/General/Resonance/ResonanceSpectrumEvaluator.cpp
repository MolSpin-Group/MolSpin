/////////////////////////////////////////////////////////////////////////
// ResonanceSpectrumEvaluator implementation (RunSection::General::Resonance)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// DEVELOPER WORKFLOW / OWNERSHIP MAP
// ----------------------------------------------------------------------
// Assembly of the final resonance spectrum.
//
// What is done here:
//   - Combines detected resonances, transition moments, field Jacobians, powder weights and analytical lineshapes.
//   - Accumulates the observable spectrum without changing the underlying eigenproblem.
//
// Connections to the General framework / SpinAPI:
//   - Coordinates GeneralResonanceHamiltonian, ResonanceTransitionDetector, ResonanceTransitionMoments, ResonanceFieldJacobian and ResonanceLineshape.
//   - SpinAPI remains responsible for spin-space operators and rotated Hamiltonians.
//
// Why this ownership is used:
//   - Spectrum assembly is kept separate from resonance finding so each stage can be validated independently.
/////////////////////////////////////////////////////////////////////////

#include "ResonanceSpectrumEvaluator.h"
#include "ResonanceFieldJacobian.h"
#include "ResonanceLineshape.h"
#include "ResonanceTransitionDetector.h"
#include "ResonanceTransitionMoments.h"

#include <cmath>
#include <vector>

namespace RunSection::General::Resonance
{
    bool ResonanceSpectrumEvaluator::Evaluate(const arma::sp_cx_mat &hamiltonian,
        const arma::cx_mat &density, const arma::sp_cx_mat &dHdB,
        const arma::cx_mat &muX, const arma::cx_mat &muY,
        const SpectrumRequest &request, SpectrumPoint &spectrum, std::string &error)
    {
        error.clear();
        spectrum = SpectrumPoint{};
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
        if (!std::isfinite(request.microwaveFrequencyGHz) || request.microwaveFrequencyGHz <= 0.0 ||
            !std::isfinite(request.linewidth_mT) || request.linewidth_mT < 0.0)
        {
            error = "invalid microwave frequency or linewidth";
            return false;
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

		// Resolve degenerate field-following states before transforming either
		// the density or magnetic-dipole operators. Otherwise populations and
		// transition moments depend on an arbitrary basis chosen by eig_sym.
		const arma::cx_mat densityEigen = eigenvectors.t() * density * eigenvectors;
        const arma::vec populations = arma::real(densityEigen.diag());
        if (!populations.is_finite())
        {
            error = "non-finite eigenbasis populations";
            return false;
        }

        std::vector<Transition> transitions;
        const double omegaMw = 2.0 * arma::datum::pi * request.microwaveFrequencyGHz;
        if (!ResonanceTransitionDetector::Detect(energies, populations, dEdB, omegaMw,
                transitions, error, request.populationThreshold, request.minimumSlope,
                request.maximumDBdOmega))
            return false;

        arma::cx_mat muXEigen, muYEigen;
        if (!ResonanceTransitionMoments::Transform(eigenvectors, muX, muY,
                muXEigen, muYEigen, error))
            return false;

        for (const auto &transition : transitions)
        {
            const double line = ResonanceLineshape::Evaluate(request.lineshape,
                transition.detuningField_mT, request.linewidth_mT);
            if (line == 0.0)
                continue;

            const double fieldWeight = transition.dBdOmega * line;
            const TransitionMoment moment = ResonanceTransitionMoments::Evaluate(
                muXEigen, muYEigen, transition.lower, transition.upper);
            const double prefactor = transition.populationDifference * fieldWeight;
            spectrum.totalX += prefactor * moment.x;
            spectrum.totalY += prefactor * moment.y;
            spectrum.totalPerpendicular += prefactor * moment.perpendicular;
            ++spectrum.acceptedTransitions;
        }

        return std::isfinite(spectrum.totalX) && std::isfinite(spectrum.totalY) &&
               std::isfinite(spectrum.totalPerpendicular);
    }
}
