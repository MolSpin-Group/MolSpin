/////////////////////////////////////////////////////////////////////////
// ResonanceTransitionMoments implementation (RunSection::General::Resonance)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "ResonanceTransitionMoments.h"

#include <cmath>

namespace RunSection::General::Resonance
{
    bool ResonanceTransitionMoments::Transform(const arma::cx_mat &eigenvectors,
        const arma::cx_mat &muX, const arma::cx_mat &muY,
        arma::cx_mat &muXEigen, arma::cx_mat &muYEigen, std::string &error)
    {
        error.clear();
        if (eigenvectors.n_rows == 0 || eigenvectors.n_rows != eigenvectors.n_cols)
        {
            error = "eigenvector matrix must be non-empty and square";
            return false;
        }
        if (muX.n_rows != eigenvectors.n_rows || muX.n_cols != eigenvectors.n_rows ||
            muY.n_rows != eigenvectors.n_rows || muY.n_cols != eigenvectors.n_rows)
        {
            error = "transition-moment operator dimension does not match the eigensystem";
            return false;
        }

        const arma::cx_mat Udag = eigenvectors.t();
        muXEigen = Udag * muX * eigenvectors;
        muYEigen = Udag * muY * eigenvectors;
        return muXEigen.is_finite() && muYEigen.is_finite();
    }

    TransitionMoment ResonanceTransitionMoments::Evaluate(const arma::cx_mat &muXEigen,
        const arma::cx_mat &muYEigen, arma::uword lower, arma::uword upper)
    {
        TransitionMoment result;
        if (lower >= muXEigen.n_rows || upper >= muXEigen.n_cols ||
            lower >= muYEigen.n_rows || upper >= muYEigen.n_cols)
            return result;

        result.x = std::norm(muXEigen(lower, upper));
        result.y = std::norm(muYEigen(lower, upper));
        result.perpendicular = 0.5 * (result.x + result.y);
        return result;
    }
}
