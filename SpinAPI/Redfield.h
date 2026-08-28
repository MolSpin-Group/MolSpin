/////////////////////////////////////////////////////////////////////////
// Redfield theory primitives (SpinAPI Module)
// ------------------
// Reusable frequency, spectral-density, and relaxation-tensor operations.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_SpinAPI_Redfield
#define MOD_SpinAPI_Redfield

#include "Relaxation.h"
#include <armadillo>
#include <string>

namespace SpinAPI::Redfield
{
    bool FrequencyMatrix(const arma::vec &_eigenvalues,
        arma::cx_mat &_frequencies, std::string *_error = nullptr);

    bool SpectralDensity(
        const std::vector<SpinAPI::Relaxation::ExponentialTerm> &_terms,
        const arma::cx_mat &_frequencies,
        SpinAPI::Relaxation::SpectralDensityFunction _function,
        arma::cx_mat &_spectralDensity, std::string *_error = nullptr);

    bool RelaxationTensor(const arma::cx_mat &_op1, const arma::cx_mat &_op2,
        const arma::cx_mat &_spectralDensity, arma::cx_mat &_tensor,
        std::string *_error = nullptr);
}

#endif
