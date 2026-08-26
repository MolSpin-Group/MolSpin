/////////////////////////////////////////////////////////////////////////
// ResonanceLineshape implementation (RunSection::General::Resonance)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "ResonanceLineshape.h"

#include <armadillo>
#include <cmath>

namespace RunSection::General::Resonance
{
    double ResonanceLineshape::Evaluate(Lineshape kind, double delta_mT, double fwhm_mT)
    {
        if (!std::isfinite(delta_mT) || !std::isfinite(fwhm_mT))
            return 0.0;

        if (fwhm_mT <= 0.0)
            return (std::abs(delta_mT) < 1.0e-12) ? 1.0 : 0.0;

        if (kind == Lineshape::Lorentzian)
        {
            const double gamma = 0.5 * fwhm_mT;
            return (1.0 / arma::datum::pi) * gamma /
                   (delta_mT * delta_mT + gamma * gamma);
        }

        const double x = delta_mT / fwhm_mT;
        const double prefactor = std::sqrt(4.0 * std::log(2.0) / arma::datum::pi) / fwhm_mT;
        return prefactor * std::exp(-4.0 * std::log(2.0) * x * x);
    }
}
