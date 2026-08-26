/////////////////////////////////////////////////////////////////////////
// GeneralResonanceHamiltonian (RunSection::General::Resonance)
// ------------------
// Thin resonance-facing adapter over the qualified General Hilbert Hamiltonian
// builder. It deliberately contains no tensor rotation physics of its own.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_Resonance_GeneralResonanceHamiltonian
#define MOD_RunSection_General_Resonance_GeneralResonanceHamiltonian

#include <armadillo>
#include <string>
#include <vector>
#include "HSHamiltonianBuilder.h"

namespace RunSection::General::Resonance
{
    class GeneralResonanceHamiltonian
    {
    public:
        GeneralResonanceHamiltonian(const HS::HSExecutionPlan &_plan, SpinAPI::SpinSpace &_space)
            : plan(_plan), space(_space) {}

        bool Build(const HS::HSOrientation &_orientation, arma::sp_cx_mat &_hamiltonian,
            std::string &_error) const;

        bool BuildFieldDerivative(const HS::HSOrientation &_orientation,
            const std::vector<std::string> &_zeemanInteractions, double _fieldMagnitudeT,
            arma::sp_cx_mat &_dHdB, std::string &_error) const;

    private:
        const HS::HSExecutionPlan &plan;
        SpinAPI::SpinSpace &space;
    };
}

#endif
