/////////////////////////////////////////////////////////////////////////
// GeneralResonanceHamiltonian implementation (RunSection::General::Resonance)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// DEVELOPER WORKFLOW / OWNERSHIP MAP
// ----------------------------------------------------------------------
// Resonance-specific Hamiltonian construction.
//
// What is done here:
//   - Builds the field/orientation-dependent Hamiltonian needed by the General resonance workflow.
//   - Keeps resonance scanning separate from generic time propagation.
//
// Connections to the General framework / SpinAPI:
//   - Delegates spin-interaction matrix construction and tensor rotation to SpinAPI.
//   - Used by ResonanceFieldJacobian, transition detection and spectrum evaluation.
//   - HSGeneral propagation uses HSHamiltonianBuilder instead; do not cross-dispatch task code.
//
// Why this ownership is used:
//   - A resonance solver repeatedly evaluates an eigenproblem as a function of field; this is algorithmically distinct from propagating rho(t).
//
// TODO:
//   - Strain distributions should enter through a shared Hamiltonian-realization layer so resonance spectra and time-domain General tasks sample the same physical parameter distributions.
/////////////////////////////////////////////////////////////////////////

#include "GeneralResonanceHamiltonian.h"

#include <cmath>

namespace RunSection::General::Resonance
{
    bool GeneralResonanceHamiltonian::Build(const HS::HSOrientation &orientation,
        arma::sp_cx_mat &hamiltonian, std::string &error) const
    {
        HS::HSHamiltonianBuilder builder(plan, space);
        return builder.BuildStatic(orientation, hamiltonian, nullptr, error);
    }

    bool GeneralResonanceHamiltonian::BuildFieldDerivative(const HS::HSOrientation &orientation,
        const std::vector<std::string> &zeemanInteractions, double fieldMagnitudeT,
        arma::sp_cx_mat &dHdB, std::string &error) const
    {
        error.clear();
        if (zeemanInteractions.empty())
        {
            error = "at least one Zeeman interaction is required for dH/dB";
            return false;
        }
        if (!std::isfinite(fieldMagnitudeT) || fieldMagnitudeT <= 0.0)
        {
            error = "field magnitude must be finite and positive for dH/dB";
            return false;
        }

        HS::HSExecutionPlan derivativePlan = plan;
        derivativePlan.hasH0List = true;
        derivativePlan.h0List = zeemanInteractions;
        derivativePlan.hasH1List = false;
        derivativePlan.h1List.clear();

        HS::HSHamiltonianBuilder builder(derivativePlan, space);
        arma::sp_cx_mat zeemanHamiltonian;
        if (!builder.BuildStatic(orientation, zeemanHamiltonian, nullptr, error))
            return false;
        dHdB = zeemanHamiltonian / fieldMagnitudeT;
        return true;
    }
}
