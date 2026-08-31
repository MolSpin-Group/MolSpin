/////////////////////////////////////////////////////////////////////////
// ResonanceTypes (RunSection::General::Resonance)
// ------------------
// Backend-neutral data structures for field-swept magnetic resonance.
// They carry only eigensystem/transition information and do not own a Task.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_Resonance_ResonanceTypes
#define MOD_RunSection_General_Resonance_ResonanceTypes

#include <armadillo>
#include <cstddef>
#include <vector>

namespace RunSection::General::Resonance
{
    enum class Lineshape
    {
        Gaussian,
        Lorentzian
    };

    struct Transition
    {
        arma::uword lower = 0;
        arma::uword upper = 0;
        double omega = 0.0;                 // rad/ns
        double detuningOmega = 0.0;         // rad/ns
        double populationDifference = 0.0;  // rho_lower-rho_upper
        double dOmegaDB = 0.0;              // rad/ns/T
        double dBdOmega = 0.0;              // T/(rad/ns)
        double detuningField_mT = 0.0;       // mT
    };

    struct TransitionMoment
    {
        double x = 0.0;
        double y = 0.0;
        double perpendicular = 0.0;
    };

    // Backend-neutral unbroadened resonance line at one field/orientation.
    // Exact and future hybrid nuclear solvers must produce this same object.
    struct ResonanceLine
    {
        arma::uword lower = 0;                 // backend/core lower-state label
        arma::uword upper = 0;                 // backend/core upper-state label
        double omega = 0.0;                    // rad/ns
        double populationDifference = 0.0;
        double dOmegaDB = 0.0;                 // rad/ns/T
        double dBdOmega = 0.0;                 // T/(rad/ns)
        TransitionMoment moment;
    };

    struct ResonanceLineSet
    {
        std::vector<ResonanceLine> lines;
    };

    struct SpectrumRequest
    {
        double microwaveFrequencyGHz = 0.0;
        double linewidth_mT = 0.0;
        Lineshape lineshape = Lineshape::Gaussian;
        double populationThreshold = 1.0e-15;
        double minimumSlope = 1.0e-15;
        double maximumDBdOmega = 1.0e5;
    };

    struct SpectrumPoint
    {
        double totalX = 0.0;
        double totalY = 0.0;
        double totalPerpendicular = 0.0;
        std::size_t acceptedTransitions = 0;
    };
}

#endif
