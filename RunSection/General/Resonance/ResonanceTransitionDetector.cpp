/////////////////////////////////////////////////////////////////////////
// ResonanceTransitionDetector implementation (RunSection::General::Resonance)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "ResonanceTransitionDetector.h"

#include <cmath>

namespace RunSection::General::Resonance
{
    bool ResonanceTransitionDetector::Detect(const arma::vec &energies, const arma::vec &populations,
        const arma::vec &dEdB, double omegaMw, std::vector<Transition> &transitions,
        std::string &error, double populationThreshold, double minimumSlope,
        double maximumDBdOmega)
    {
        error.clear();
        transitions.clear();
        if (energies.n_elem == 0 || populations.n_elem != energies.n_elem || dEdB.n_elem != energies.n_elem)
        {
            error = "energy, population and dE/dB vectors must have the same non-zero length";
            return false;
        }
        if (!energies.is_finite() || !populations.is_finite() || !dEdB.is_finite() || !std::isfinite(omegaMw))
        {
            error = "resonance transition inputs contain non-finite values";
            return false;
        }
        if (populationThreshold < 0.0 || minimumSlope < 0.0 || maximumDBdOmega <= 0.0)
        {
            error = "invalid resonance transition thresholds";
            return false;
        }

        for (arma::uword m = 0; m < energies.n_elem; ++m)
        {
            for (arma::uword n = m + 1; n < energies.n_elem; ++n)
            {
                const double population = populations(m) - populations(n);
                if (std::abs(population) < populationThreshold)
                    continue;

                const double slope = std::abs(dEdB(n) - dEdB(m));
                if (!std::isfinite(slope) || slope < minimumSlope)
                    continue;

                const double dBdOmega = 1.0 / slope;
                if (!std::isfinite(dBdOmega) || dBdOmega > maximumDBdOmega)
                    continue;

                Transition t;
                t.lower = m;
                t.upper = n;
                t.omega = energies(n) - energies(m);
                t.detuningOmega = t.omega - omegaMw;
                t.populationDifference = population;
                t.dOmegaDB = slope;
                t.dBdOmega = dBdOmega;
                t.detuningField_mT = 1.0e3 * t.detuningOmega * dBdOmega;
                transitions.push_back(t);
            }
        }
        return true;
    }
}
