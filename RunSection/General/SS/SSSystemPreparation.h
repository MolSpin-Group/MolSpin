/////////////////////////////////////////////////////////////////////////
// SSSystemPreparation (RunSection::General::SS)
// ------------------
// Complete one-system superspace state/generator assembly.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_SS_SSSystemPreparation
#define MOD_RunSection_General_SS_SSSystemPreparation
#include "SSExecutionPlan.h"
#include "SSOrientationSampler.h"
#include "SSLiouvillianBuilder.h"
#include <armadillo>
#include <string>
namespace RunSection::General::SS
{
    struct SSPreparedCalculation
    {
        SSPreparedSystem local;
        arma::sp_cx_mat generator;
        arma::cx_vec initialState;
        arma::cx_vec traceFunctional;
    };
    class SSSystemPreparation
    {
    public:
        static bool Prepare(const SpinAPI::system_ptr &_system,const SSExecutionPlan &_plan,
            const SSOrientation &_orientation,SSPreparedCalculation &_prepared,std::string &_error);
    };
}
#endif
