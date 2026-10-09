/////////////////////////////////////////////////////////////////////////
// Redfield theory primitives (SpinAPI Module)
// ------------------
// These routines implement the established MolSpin Redfield contraction so
// tasks can share one tested theory implementation.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "Redfield.h"

namespace SpinAPI::Redfield
{
    namespace
    {
        bool Fail(std::string *error, const std::string &message)
        {
            if (error) *error = message;
            return false;
        }
    }

    bool FrequencyMatrix(const arma::vec &eigenvalues,
        arma::cx_mat &frequencies, std::string *error)
    {
        if (error) error->clear();
        if (eigenvalues.empty() || !eigenvalues.is_finite())
            return Fail(error, "Redfield frequencies require finite Hamiltonian eigenvalues");
        frequencies.set_size(eigenvalues.n_elem, eigenvalues.n_elem);
        for (arma::uword first = 0; first < eigenvalues.n_elem; ++first)
            for (arma::uword second = 0; second < eigenvalues.n_elem; ++second)
                frequencies(first, second) = eigenvalues(first) - eigenvalues(second);
        return true;
    }

    bool SpectralDensity(
        const std::vector<SpinAPI::Relaxation::ExponentialTerm> &terms,
        const arma::cx_mat &frequencies,
        SpinAPI::Relaxation::SpectralDensityFunction function,
        arma::cx_mat &spectralDensity, std::string *error)
    {
        return SpinAPI::Relaxation::EvaluateSpectralDensity(
            terms, frequencies, function, false, spectralDensity, error);
    }

    bool RelaxationTensor(const arma::cx_mat &op1, const arma::cx_mat &op2,
        const arma::cx_mat &spectralDensity, arma::cx_mat &tensor,
        std::string *error)
    {
        if (error) error->clear();
        if (op1.n_rows == 0 || op1.n_rows != op1.n_cols ||
            arma::size(op1) != arma::size(op2) || arma::size(op1) != arma::size(spectralDensity))
            return Fail(error, "Redfield operators and spectral density must be equally sized square matrices");
        const arma::cx_mat identity = arma::eye<arma::cx_mat>(arma::size(op1));
        const arma::cx_mat weighted = op2.t() % spectralDensity.st();
        tensor = arma::kron(op1.t(), arma::conj(weighted))
            + arma::kron(weighted, op1.st())
            - arma::kron(op1 * weighted, identity)
            - arma::kron(identity, arma::conj(op1 * (op2.t() % spectralDensity.st())));
        return tensor.is_finite();
    }
}
