/////////////////////////////////////////////////////////////////////////
// NakajimaZwanzig (SpinAPI Module)
// ------------------
// Reusable low-level builders for the established MolSpin Markovian NZ form.
//
// THEORY CONTRACT
//   This file extracts algebra from the published superspace task without
//   silently changing the theory.  In the implemented formulation,
//   the spectral density is built from transition frequencies of H only and
//   the reaction operator is added separately to the final generator.
//
//   The formal second-order reactive Nakajima-Zwanzig derivation propagates the
//   kernel with a reference Liouvillian L0 that may include reaction loss:
//       T. P. Fay, L. P. Lindoy, D. E. Manolopoulos,
//       J. Chem. Phys. 154, 084121 (2021), DOI: 10.1063/5.0040519.
//
//   A future reactive-L0 implementation must be a separate physical model and
//   must not alter numerical parity with the published MolSpin workflow used in
//       DOI: 10.1021/jacs.5c06173.
//
// HIERARCHY
//   SpinAPI::NakajimaZwanzig provides theory algebra only. General/SS may use
//   it to construct a local relaxation superoperator. General/MultiSS must not
//   reimplement the tensor, and TaskMultiSSGeneral must not contain NZ algebra.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_SpinAPI_NakajimaZwanzig
#define MOD_SpinAPI_NakajimaZwanzig

#include "Relaxation.h"
#include <armadillo>
#include <complex>
#include <string>
#include <vector>

namespace SpinAPI::NakajimaZwanzig
{
    bool FrequencyMatrix(const arma::vec &_eigenvalues,
        arma::cx_mat &_omega, std::string *_error=nullptr);

    bool ExponentialSpectralDensity(const std::complex<double> &_amplitude,
        const std::complex<double> &_tauC, const arma::cx_mat &_omega,
        arma::cx_mat &_spectralDensity, std::string *_error=nullptr);

    bool MultiExponentialSpectralDensity(const std::vector<double> &_amplitudes,
        const std::vector<double> &_tauC, const arma::cx_mat &_omega,
        arma::cx_mat &_spectralDensity, std::string *_error=nullptr);

    bool SpectralDensity(
        const std::vector<SpinAPI::Relaxation::ExponentialTerm> &_terms,
        const arma::cx_mat &_omega, arma::cx_mat &_spectralDensity,
        std::string *_error=nullptr);

    bool RelaxationTensor(const arma::cx_mat &_op1,const arma::cx_mat &_op2,
        const arma::cx_mat &_spectralDensity,arma::cx_mat &_tensor,
        std::string *_error=nullptr);
}

#endif
