/////////////////////////////////////////////////////////////////////////
// NakajimaZwanzig (SpinAPI Module)
// ------------------
// Reusable low-level builders for the *historical MolSpin Markovian NZ form*.
//
// MIGRATION POLICY
//   This file extracts algebra from the published/historical superspace task
//   without silently changing the theory.  In that historical implementation
//   the spectral density is built from transition frequencies of H only and
//   the reaction operator is added separately to the final generator.
//
//   The formal second-order reactive Nakajima-Zwanzig derivation propagates the
//   kernel with a reference Liouvillian L0 that may include reaction loss:
//       T. P. Fay, L. P. Lindoy, D. E. Manolopoulos,
//       J. Chem. Phys. 154, 084121 (2021), DOI: 10.1063/5.0040519.
//
//   Therefore `Historical...` in these API names is intentional.  A future
//   reactive-L0 implementation must be a separately named variant and must not
//   alter numerical parity with the published MolSpin workflow used in
//       DOI: 10.1021/jacs.5c06173.
//
// HIERARCHY
//   SpinAPI::NakajimaZwanzig provides theory algebra only. General/SS may use
//   it to construct a local relaxation superoperator. General/MultiSS must not
//   reimplement the tensor, and TaskMultiSSGeneral must not contain NZ algebra.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_SpinAPI_NakajimaZwanzig
#define MOD_SpinAPI_NakajimaZwanzig

#include <armadillo>
#include <complex>
#include <string>
#include <vector>

namespace SpinAPI::NakajimaZwanzig
{
    bool HistoricalFrequencyMatrix(const arma::vec &_eigenvalues,
        arma::cx_mat &_omega, std::string *_error=nullptr);

    bool HistoricalExponentialSpectralDensity(const std::complex<double> &_amplitude,
        const std::complex<double> &_tauC, const arma::cx_mat &_omega,
        arma::cx_mat &_spectralDensity, std::string *_error=nullptr);

    bool HistoricalMultiExponentialSpectralDensity(const std::vector<double> &_amplitudes,
        const std::vector<double> &_tauC, const arma::cx_mat &_omega,
        arma::cx_mat &_spectralDensity, std::string *_error=nullptr);

    bool HistoricalTensor(const arma::cx_mat &_op1,const arma::cx_mat &_op2,
        const arma::cx_mat &_spectralDensity,arma::cx_mat &_tensor,
        std::string *_error=nullptr);
}

#endif
