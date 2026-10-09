/////////////////////////////////////////////////////////////////////////
// ResonanceTransitionMoments implementation (RunSection::General::Resonance)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// DEVELOPER WORKFLOW / OWNERSHIP MAP
// ----------------------------------------------------------------------
// Resonance transition-moment evaluation.
//
// What is done here:
//   - Evaluates matrix elements of the excitation/detection operator between Hamiltonian eigenstates.
//   - Produces transition strengths used by the spectrum evaluator.
//
// Connections to the General framework / SpinAPI:
//   - Uses SpinAPI spin-space operators and eigenvectors from the resonance Hamiltonian.
//   - Does not determine resonance fields or apply lineshape broadening.
//
// Why this ownership is used:
//   - Selection rules/intensities are quantum-mechanical operator matrix elements and should not be mixed with numerical crossing detection.
/////////////////////////////////////////////////////////////////////////

#include "ResonanceTransitionMoments.h"

#include <cmath>

namespace RunSection::General::Resonance
{
    bool ResonanceTransitionMoments::Transform(const arma::cx_mat &eigenvectors,
        const arma::cx_mat &muX, const arma::cx_mat &muY,
        arma::cx_mat &muXEigen, arma::cx_mat &muYEigen, std::string &error)
    {
        error.clear();
        if (eigenvectors.n_rows == 0 || eigenvectors.n_rows != eigenvectors.n_cols)
        {
            error = "eigenvector matrix must be non-empty and square";
            return false;
        }
        if (muX.n_rows != eigenvectors.n_rows || muX.n_cols != eigenvectors.n_rows ||
            muY.n_rows != eigenvectors.n_rows || muY.n_cols != eigenvectors.n_rows)
        {
            error = "transition-moment operator dimension does not match the eigensystem";
            return false;
        }

        const arma::cx_mat Udag = eigenvectors.t();
        muXEigen = Udag * muX * eigenvectors;
        muYEigen = Udag * muY * eigenvectors;
        return muXEigen.is_finite() && muYEigen.is_finite();
    }

    TransitionMoment ResonanceTransitionMoments::Evaluate(const arma::cx_mat &muXEigen,
        const arma::cx_mat &muYEigen, arma::uword lower, arma::uword upper)
    {
        TransitionMoment result;
        if (lower >= muXEigen.n_rows || upper >= muXEigen.n_cols ||
            lower >= muYEigen.n_rows || upper >= muYEigen.n_cols)
            return result;

        result.x = std::norm(muXEigen(lower, upper));
        result.y = std::norm(muYEigen(lower, upper));
        result.perpendicular = 0.5 * (result.x + result.y);
        return result;
    }
    bool ResonanceTransitionMoments::TransformChannels(
        const arma::cx_mat &eigenvectors,
        const std::vector<ResonanceDetectionOperator> &channels,
        std::vector<ResonanceDetectionOperator> &channelsEigen,
        std::string &error)
    {
        error.clear();
        channelsEigen.clear();
        channelsEigen.reserve(channels.size());

        for (const auto &channel:channels)
        {
            ResonanceDetectionOperator transformed;
            if (!Transform(
                    eigenvectors,
                    channel.x,channel.y,
                    transformed.x,transformed.y,
                    error))
                return false;
            channelsEigen.push_back(transformed);
        }
        return true;
    }

    bool ResonanceTransitionMoments::EvaluateResolved(
        const arma::cx_mat &muXEigen,
        const arma::cx_mat &muYEigen,
        const std::vector<ResonanceDetectionOperator> &channelsEigen,
        arma::uword lower, arma::uword upper,
        TransitionMoment &result,std::string &error)
    {
        error.clear();
        result=TransitionMoment{};

        if (lower >= muXEigen.n_rows ||
            upper >= muXEigen.n_cols ||
            lower >= muYEigen.n_rows ||
            upper >= muYEigen.n_cols)
        {
            error =
                "resolved transition-moment index is outside the total operator";
            return false;
        }

        result=Evaluate(
            muXEigen,muYEigen,lower,upper);
        result.channels.reserve(
            channelsEigen.size());

        double incoherentX=0.0;
        double incoherentY=0.0;
        const arma::cx_double I(0.0,1.0);

        for (const auto &channel:channelsEigen)
        {
            if (channel.x.n_rows!=muXEigen.n_rows ||
                channel.x.n_cols!=muXEigen.n_cols ||
                channel.y.n_rows!=muYEigen.n_rows ||
                channel.y.n_cols!=muYEigen.n_cols ||
                !channel.x.is_finite() ||
                !channel.y.is_finite())
            {
                error =
                    "resolved transition-moment channel dimensions do not match the total operator";
                result=TransitionMoment{};
                return false;
            }

            const arma::cx_double mux=
                channel.x(lower,upper);
            const arma::cx_double muy=
                channel.y(lower,upper);

            TransitionMomentChannel resolved;
            resolved.x=std::norm(mux);
            resolved.y=std::norm(muy);
            resolved.perpendicular=
                0.5*(resolved.x+resolved.y);
            resolved.plus=
                std::norm(mux+I*muy);
            resolved.minus=
                std::norm(mux-I*muy);

            incoherentX+=resolved.x;
            incoherentY+=resolved.y;
            result.channels.push_back(resolved);
        }

        result.crossX=result.x-incoherentX;
        result.crossY=result.y-incoherentY;

        if (!IsFinite(result))
        {
            error =
                "resolved transition moment contains non-finite values";
            result=TransitionMoment{};
            return false;
        }
        return true;
    }

    TransitionMoment ResonanceTransitionMoments::Scale(
        const TransitionMoment &moment,double factor)
    {
        TransitionMoment result=moment;
        result.x*=factor;
        result.y*=factor;
        result.perpendicular*=factor;
        result.crossX*=factor;
        result.crossY*=factor;

        for (auto &channel:result.channels)
        {
            channel.x*=factor;
            channel.y*=factor;
            channel.perpendicular*=factor;
            channel.plus*=factor;
            channel.minus*=factor;
        }
        return result;
    }

    bool ResonanceTransitionMoments::IsFinite(
        const TransitionMoment &moment)
    {
        if (!std::isfinite(moment.x) ||
            !std::isfinite(moment.y) ||
            !std::isfinite(moment.perpendicular) ||
            !std::isfinite(moment.crossX) ||
            !std::isfinite(moment.crossY))
            return false;

        for (const auto &channel:moment.channels)
        {
            if (!std::isfinite(channel.x) ||
                !std::isfinite(channel.y) ||
                !std::isfinite(channel.perpendicular) ||
                !std::isfinite(channel.plus) ||
                !std::isfinite(channel.minus))
                return false;
        }
        return true;
    }

}
