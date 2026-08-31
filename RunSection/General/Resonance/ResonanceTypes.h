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

    // One transverse detection operator before transformation to the
    // instantaneous Hamiltonian eigenbasis. The ordered channel list is owned
    // by the caller, normally one channel per resolved detection spin.
    struct ResonanceDetectionOperator
    {
        arma::cx_mat x;
        arma::cx_mat y;
    };

    // Intensity decomposition for one resolved detection channel. Circular
    // channels follow the historical StaticHS-Resonance-Spectra convention:
    // plus=|mu_x+i mu_y|^2 and minus=|mu_x-i mu_y|^2.
    struct TransitionMomentChannel
    {
        double x = 0.0;
        double y = 0.0;
        double perpendicular = 0.0;
        double plus = 0.0;
        double minus = 0.0;
    };

    struct TransitionMoment
    {
        // Coherent total intensities.
        double x = 0.0;
        double y = 0.0;
        double perpendicular = 0.0;

        // Coherent total minus the incoherent channel sum. These reproduce the
        // historical cross_x/cross_y definition used by StaticHS resonance.
        double crossX = 0.0;
        double crossY = 0.0;

        // Ordered per-detection-channel intensities.
        std::vector<TransitionMomentChannel> channels;
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

        // A line provider may know transition frequencies/intensities before a
        // complete field derivative has been qualified. Spectrum evaluation
        // must reject such a set rather than silently use an approximate
        // dB/domega. ExactResonanceSolver always sets this true.
        bool fieldJacobianQualified = false;
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
