/////////////////////////////////////////////////////////////////////////
// ResonanceFieldJacobian (RunSection::General::Resonance)
// ------------------
// Converts dH/dB into eigenstate energy slopes dE_n/dB. The resulting
// transition slope supplies the field-frequency Jacobian used in field sweeps.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_Resonance_ResonanceFieldJacobian
#define MOD_RunSection_General_Resonance_ResonanceFieldJacobian

#include <armadillo>
#include <string>

namespace RunSection::General::Resonance
{
    class ResonanceFieldJacobian
    {
    public:
        static bool DiagonalEnergyDerivatives(const arma::cx_mat &_eigenvectors,
            const arma::sp_cx_mat &_dHdB, arma::vec &_dEdB, std::string &_error);
    };
}

#endif
