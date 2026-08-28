/////////////////////////////////////////////////////////////////////////
// SSNakajimaZwanzigBuilder (RunSection::General::SS)
// ------------------
// Local one-manifold assembly of the established MolSpin/JACS Markovian
// Nakajima-Zwanzig relaxation model.  The low-level tensor algebra remains
// owned by SpinAPI::NakajimaZwanzig.  This layer interprets the established
// Interaction properties and constructs the local superspace contribution.
//
// IMPORTANT THEORY CONTRACT
//   * transition frequencies are obtained from the static Hamiltonian H only;
//   * reaction loss is NOT included in the NZ reference propagator;
//   * reaction/transfer is added separately by SSGeneral/MultiSSGeneral;
//   * this reproduces the published MolSpin workflow used for
//       DOI: 10.1021/jacs.5c06173
//     and is deliberately distinct from a future reactive-L0 NZ model.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_SS_SSNakajimaZwanzigBuilder
#define MOD_RunSection_General_SS_SSNakajimaZwanzigBuilder

#include <armadillo>
#include <string>
#include "SpinAPIfwd.h"

namespace SpinAPI { class SpinSpace; }

namespace RunSection::General::SS
{
    class SSNakajimaZwanzigBuilder
    {
    public:
        // Build and sum all NZ-enabled Interaction contributions.  An
        // interaction is NZ-enabled by carrying the established `g` and
        // `tau_c` properties.  The current General migration supports the
        // published/simple exponential-list input and Cartesian or rank-0/2
        // spherical operator bases. Matrix-valued multi-exponential fits use
        // the established ordered Cartesian correlation-channel layout. The
        // frequency-domain tensor is assembled in the Hamiltonian eigenbasis
        // and returned in the caller's propagation basis.
        static bool Build(const SpinAPI::system_ptr &_system,
            SpinAPI::SpinSpace &_space, const arma::sp_cx_mat &_hamiltonian,
            const arma::mat &_molecularToLab,
            arma::sp_cx_mat &_relaxation, std::string &_error);
    };
}
#endif
