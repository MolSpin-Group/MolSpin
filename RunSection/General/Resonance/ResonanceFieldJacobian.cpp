/////////////////////////////////////////////////////////////////////////
// ResonanceFieldJacobian implementation (RunSection::General::Resonance)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "ResonanceFieldJacobian.h"

#include <cmath>

namespace RunSection::General::Resonance
{
    bool ResonanceFieldJacobian::DiagonalEnergyDerivatives(const arma::cx_mat &eigenvectors,
        const arma::sp_cx_mat &dHdB, arma::vec &dEdB, std::string &error)
    {
        error.clear();
        if (eigenvectors.n_rows == 0 || eigenvectors.n_rows != eigenvectors.n_cols)
        {
            error = "eigenvector matrix must be non-empty and square";
            return false;
        }
        if (dHdB.n_rows != eigenvectors.n_rows || dHdB.n_cols != eigenvectors.n_rows)
        {
            error = "dH/dB dimension does not match the eigensystem";
            return false;
        }

        const arma::cx_mat derivativeVectors = dHdB * eigenvectors;
        dEdB.set_size(eigenvectors.n_cols);
        for (arma::uword n = 0; n < eigenvectors.n_cols; ++n)
        {
            const double value = std::real(arma::cdot(eigenvectors.col(n), derivativeVectors.col(n)));
            if (!std::isfinite(value))
            {
                error = "non-finite eigenstate field derivative";
                dEdB.reset();
                return false;
            }
            dEdB(n) = value;
        }
        return true;
    }
}
