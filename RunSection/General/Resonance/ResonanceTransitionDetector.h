/////////////////////////////////////////////////////////////////////////
// ResonanceTransitionDetector (RunSection::General::Resonance)
// ------------------
// Enumerates population-carrying resonances and converts frequency detuning
// to field detuning using the local transition slope.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_Resonance_ResonanceTransitionDetector
#define MOD_RunSection_General_Resonance_ResonanceTransitionDetector

#include "ResonanceTypes.h"
#include <armadillo>
#include <string>
#include <vector>

namespace RunSection::General::Resonance
{
    class ResonanceTransitionDetector
    {
    public:
        static bool Detect(const arma::vec &_energies, const arma::vec &_populations,
            const arma::vec &_dEdB, double _omegaMw,
            std::vector<Transition> &_transitions, std::string &_error,
            double _populationThreshold = 1.0e-15,
            double _minimumSlope = 1.0e-15,
            double _maximumDBdOmega = 1.0e5);
    };
}

#endif
