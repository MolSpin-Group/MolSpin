/////////////////////////////////////////////////////////////////////////
// SSRedfieldBuilder (RunSection::General::SS)
// ------------------
// Local one-manifold assembly of the established MolSpin Redfield model.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_SS_SSRedfieldBuilder
#define MOD_RunSection_General_SS_SSRedfieldBuilder

#include "SpinAPIfwd.h"
#include <armadillo>
#include <string>

namespace SpinAPI { class SpinSpace; }

namespace RunSection::General::SS
{
    class SSRedfieldBuilder
    {
    public:
        static bool Build(const SpinAPI::system_ptr &_system,
            SpinAPI::SpinSpace &_space, const arma::sp_cx_mat &_hamiltonian,
            const arma::mat &_molecularToLab,
            arma::sp_cx_mat &_relaxation, std::string &_error);
    };
}

#endif
