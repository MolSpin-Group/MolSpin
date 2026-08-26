/////////////////////////////////////////////////////////////////////////
// ResonanceSpectrumEvaluator (RunSection::General::Resonance)
// ------------------
// Backend-neutral field-swept resonance evaluator. Hamiltonian construction,
// state preparation and magnetic-dipole operator construction remain external;
// this class owns only eigensystem-to-spectrum physics.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_Resonance_ResonanceSpectrumEvaluator
#define MOD_RunSection_General_Resonance_ResonanceSpectrumEvaluator

#include "ResonanceTypes.h"
#include <armadillo>
#include <string>

namespace RunSection::General::Resonance
{
    class ResonanceSpectrumEvaluator
    {
    public:
        static bool Evaluate(const arma::sp_cx_mat &_hamiltonian,
            const arma::cx_mat &_density,
            const arma::sp_cx_mat &_dHdB,
            const arma::cx_mat &_muX,
            const arma::cx_mat &_muY,
            const SpectrumRequest &_request,
            SpectrumPoint &_spectrum,
            std::string &_error);
    };
}

#endif
