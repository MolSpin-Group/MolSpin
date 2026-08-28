/////////////////////////////////////////////////////////////////////////
// QuantumMap (SpinAPI Module)
// ------------------
// Reusable instantaneous completely-positive map primitives.
//
// In the MultiSS hierarchy a QuantumMap is used by MultiSSEventController for
// discontinuous events.  It does NOT implement a task scheduler and it does
// not inspect PulseSequence mutable Active flags.
//
// For an incoherent instantaneous transfer event with probability f on source
// support G and isometry C, the represented block-diagonal map is
//
//   rho_s' = K0 rho_s K0^dagger,
//   rho_t' = rho_t + f C rho_s C^dagger,
//   K0 = I - (1-sqrt(1-f)) G,
//
// when G is a projector.  The construction keeps source coherences with the
// untransferred subspace with the correct sqrt(1-f) amplitude.  The idealized
// delta-pump/push treatment used for spin-selective radical-pair experiments
// motivates this event limit:
//      DOI: 10.1126/science.abl4254
//
// A finite pulse should instead be represented by a TimeProfile multiplying a
// TransferChannel.  The two descriptions converge in the short-pulse limit
// when the integrated transfer action is held fixed.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_SpinAPI_QuantumMap
#define MOD_SpinAPI_QuantumMap

#include <armadillo>
#include <string>

#include "TransferChannel.h"

namespace SpinAPI
{
    class QuantumMap
    {
    public:
        static bool ApplyTransferEvent(const TransferChannel &_channel,
            double _fraction, arma::cx_mat &_sourceDensity,
            arma::cx_mat &_targetDensity, std::string &_error,
            double _tolerance = 1.0e-10);

        // Lower-level overload used after General/MultiSS has rotated a channel
        // into the current common crystallite orientation.
        static bool ApplyTransferEvent(const arma::sp_cx_mat &_sourceEffect,
            const std::vector<arma::sp_cx_mat> &_kraus, double _fraction,
            arma::cx_mat &_sourceDensity, arma::cx_mat &_targetDensity,
            std::string &_error, double _tolerance = 1.0e-10);
    };
}

#endif
