/////////////////////////////////////////////////////////////////////////
// ResonanceTransitionMoments (RunSection::General::Resonance)
// ------------------
// Transforms user/backend-provided transverse magnetic-dipole operators to the
// instantaneous energy eigenbasis and evaluates Cartesian/perpendicular moments.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_Resonance_ResonanceTransitionMoments
#define MOD_RunSection_General_Resonance_ResonanceTransitionMoments

#include "ResonanceTypes.h"
#include <armadillo>
#include <string>

namespace RunSection::General::Resonance
{
    class ResonanceTransitionMoments
    {
    public:
        static bool Transform(const arma::cx_mat &_eigenvectors,
            const arma::cx_mat &_muX, const arma::cx_mat &_muY,
            arma::cx_mat &_muXEigen, arma::cx_mat &_muYEigen,
            std::string &_error);

        static TransitionMoment Evaluate(const arma::cx_mat &_muXEigen,
            const arma::cx_mat &_muYEigen, arma::uword _lower, arma::uword _upper);

        static bool TransformChannels(
            const arma::cx_mat &_eigenvectors,
            const std::vector<ResonanceDetectionOperator> &_channels,
            std::vector<ResonanceDetectionOperator> &_channelsEigen,
            std::string &_error);

        static bool EvaluateResolved(
            const arma::cx_mat &_muXEigen,
            const arma::cx_mat &_muYEigen,
            const std::vector<ResonanceDetectionOperator> &_channelsEigen,
            arma::uword _lower, arma::uword _upper,
            TransitionMoment &_result,std::string &_error);

        static TransitionMoment Scale(
            const TransitionMoment &_moment,double _factor);

        static bool IsFinite(const TransitionMoment &_moment);
    };
}

#endif
