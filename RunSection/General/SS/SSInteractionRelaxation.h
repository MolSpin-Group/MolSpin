/////////////////////////////////////////////////////////////////////////
// SSInteractionRelaxation class (RunSection::General::SS)
// ------------------
// Shared parsing and operator construction for interaction-derived
// Redfield and Nakajima-Zwanzig relaxation.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_SS_SSInteractionRelaxation
#define MOD_RunSection_General_SS_SSInteractionRelaxation

#include "Relaxation.h"
#include "SpinAPIfwd.h"
#include <armadillo>
#include <string>
#include <vector>

namespace SpinAPI { class SpinSpace; }

namespace RunSection::General::SS
{
    class SSInteractionRelaxation
    {
    public:
        static bool HasCorrelationInput(const SpinAPI::interaction_ptr &_interaction,
            bool &_hasInput, std::string &_error);

        static bool BuildCorrelationExpansion(
            const SpinAPI::interaction_ptr &_interaction,
            std::size_t _operatorCount, int _terms,
            SpinAPI::Relaxation::CorrelationExpansion &_expansion,
            std::string &_error);

        static bool BuildOperatorBasis(SpinAPI::SpinSpace &_space,
            const SpinAPI::interaction_ptr &_interaction,
            const SpinAPI::spin_ptr &_spin1, const SpinAPI::spin_ptr &_spin2,
            int _opsMode, const arma::mat &_molecularToLab,
            std::vector<arma::cx_mat> &_operators,
            std::string &_error);
    };
}

#endif
