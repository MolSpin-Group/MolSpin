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
//   - Consumes backend-neutral unbroadened resonance lines.
//   - Applies microwave detuning, field Jacobians and analytical lineshapes.
//   - Retains the historical matrix API as an exact-backend compatibility path.
//
// Connections to the General framework / SpinAPI:
//   - ExactResonanceSolver currently provides the qualified full-Hilbert lines.
//   - Future hybrid nuclear solvers must provide the same ResonanceLineSet.
//   - Hamiltonian/tensor physics remains in GeneralResonanceHamiltonian/SpinAPI.
//
// Why this ownership is used:
//   - Spectrum assembly must not know whether nuclei were represented exactly
//     or perturbatively. Keeping solver physics upstream prevents duplicate
//     powder/lineshape implementations and is the R1 seam for hybrid EPR.
//
// TODO:
//   - Add HybridNuclearResonanceSolver only after the exact line-backend seam is
//     numerically certified against the frozen legacy resonance tests.
/////////////////////////////////////////////////////////////////////////

#include "ResonanceSpectrumEvaluator.h"
#include "ExactResonanceSolver.h"
#include "ResonanceLineshape.h"
#include "ResonanceTransitionMoments.h"

#include <cmath>

namespace RunSection::General::Resonance
{
    namespace
    {
        bool ValidateRequest(const SpectrumRequest &request, std::string &error)
        {
            if (!std::isfinite(request.microwaveFrequencyGHz) ||
                request.microwaveFrequencyGHz <= 0.0 ||
                !std::isfinite(request.linewidth_mT) ||
                request.linewidth_mT < 0.0)
            {
                error = "invalid microwave frequency or linewidth";
                return false;
            }
            return true;
        }
    }

    bool ResonanceSpectrumEvaluator::Evaluate(const arma::sp_cx_mat &hamiltonian,
        const arma::cx_mat &density, const arma::sp_cx_mat &dHdB,
        const arma::cx_mat &muX, const arma::cx_mat &muY,
        const SpectrumRequest &request, SpectrumPoint &spectrum, std::string &error)
    {
        error.clear();
        spectrum = SpectrumPoint{};

        // Preserve the pre-R1 input-validation ordering of the public exact API.
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
        if (!ValidateRequest(request, error))
            return false;

        ResonanceLineSet lines;
        if (!ExactResonanceSolver::Generate(
                hamiltonian, density, dHdB, muX, muY, request, lines, error))
            return false;

        return Evaluate(lines, request, spectrum, error);
    }

    bool ResonanceSpectrumEvaluator::Evaluate(
        const ResonanceLineSet &lines,
        const SpectrumRequest &request,
        SpectrumPoint &spectrum,
        std::string &error)
    {
        error.clear();
        spectrum=SpectrumPoint{};

        if (!ValidateRequest(request,error))
            return false;
        if (!lines.fieldJacobianQualified)
        {
            error =
                "resonance line set does not have a qualified field Jacobian";
            return false;
        }

        const std::size_t channelCount=
            lines.lines.empty()
            ? 0
            : lines.lines.front().moment.channels.size();
        spectrum.channels.assign(
            channelCount,TransitionMomentChannel{});

        const double omegaMw=
            2.0*arma::datum::pi*
            request.microwaveFrequencyGHz;

        for (const auto &line:lines.lines)
        {
            if (!std::isfinite(line.omega) ||
                !std::isfinite(line.populationDifference) ||
                !std::isfinite(line.dOmegaDB) ||
                !std::isfinite(line.dBdOmega) ||
                !ResonanceTransitionMoments::IsFinite(
                    line.moment))
            {
                error =
                    "resonance line set contains non-finite values";
                spectrum=SpectrumPoint{};
                return false;
            }
            if (line.moment.channels.size()!=channelCount)
            {
                error =
                    "resonance line set mixes resolved detection-channel cardinalities";
                spectrum=SpectrumPoint{};
                return false;
            }

            const double detuningOmega=
                line.omega-omegaMw;
            const double detuningField_mT=
                1.0e3*detuningOmega*
                line.dBdOmega;
            const double profile=
                ResonanceLineshape::Evaluate(
                    request.lineshape,
                    detuningField_mT,
                    request.linewidth_mT);
            if (profile==0.0)
                continue;

            const double fieldWeight=
                line.dBdOmega*profile;
            const double prefactor=
                line.populationDifference*
                fieldWeight;

            spectrum.totalX+=
                prefactor*line.moment.x;
            spectrum.totalY+=
                prefactor*line.moment.y;
            spectrum.totalPerpendicular+=
                prefactor*
                line.moment.perpendicular;
            spectrum.crossX+=
                prefactor*line.moment.crossX;
            spectrum.crossY+=
                prefactor*line.moment.crossY;

            for (std::size_t k=0;
                 k<channelCount;++k)
            {
                const auto &source=
                    line.moment.channels[k];
                auto &target=
                    spectrum.channels[k];

                target.x+=prefactor*source.x;
                target.y+=prefactor*source.y;
                target.perpendicular+=
                    prefactor*source.perpendicular;
                target.plus+=
                    prefactor*source.plus;
                target.minus+=
                    prefactor*source.minus;
            }

            ++spectrum.acceptedTransitions;
        }

        const auto finiteChannel=
            [](const TransitionMomentChannel &channel)
            {
                return
                    std::isfinite(channel.x) &&
                    std::isfinite(channel.y) &&
                    std::isfinite(channel.perpendicular) &&
                    std::isfinite(channel.plus) &&
                    std::isfinite(channel.minus);
            };

        if (!std::isfinite(spectrum.totalX) ||
            !std::isfinite(spectrum.totalY) ||
            !std::isfinite(
                spectrum.totalPerpendicular) ||
            !std::isfinite(spectrum.crossX) ||
            !std::isfinite(spectrum.crossY))
            return false;

        for (const auto &channel:spectrum.channels)
        {
            if (!finiteChannel(channel))
                return false;
        }
        return true;
    }
}
