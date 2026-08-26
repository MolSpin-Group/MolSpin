/////////////////////////////////////////////////////////////////////////
// GeneralResonanceHamiltonian implementation (RunSection::General::Resonance)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
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
