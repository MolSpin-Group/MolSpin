/////////////////////////////////////////////////////////////////////////
// SSExecutionPlan (RunSection::General::SS)
// ------------------
// Resolved calculation policy for one-SpinSystem Liouville/superspace dynamics.
// Physics construction is delegated to the remaining General/SS components.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_SS_SSExecutionPlan
#define MOD_RunSection_General_SS_SSExecutionPlan

#include "MSDParserfwd.h"
#include "PowderGrid.h"
#include "SpinAPIDefines.h"
#include "SSLiouvillianBuilder.h"
#include <string>

namespace RunSection::General::SS
{
    enum class SSCalculation { TimeEvolution, TimeIntegrated, SteadyState };
    enum class SSPropagation { Exponential, RK4 };
    enum class SSObservableMode { States, TransitionYields, Both };
    enum class SSOrientationMode { Identity, Explicit, Powder2D, PowderSO3 };
    struct SSExecutionPlan
    {
        SSCalculation calculation = SSCalculation::TimeIntegrated;
        SSPropagation propagation = SSPropagation::Exponential;
        SSObservableMode observables = SSObservableMode::States;
        SSOrientationMode orientation = SSOrientationMode::Identity;
        SSHamiltonianMode hamiltonianMode = SSHamiltonianMode::FixedFull;
        SpinAPI::ReactionOperatorType reactionOperator = SpinAPI::ReactionOperatorType::Haberkorn;
        SSRelaxationModel relaxationModel = SSRelaxationModel::Operators;

        double totalTime = 1000.0;
        double timeStep = 1.0;
        double solverResidualTolerance = 1.0e-8;
        bool diagnostics = true;

        SpinAPI::PowderGridType powderGridType = SpinAPI::PowderGridType::Uniform;
        SpinAPI::PowderGridDomain powderDomain = SpinAPI::PowderGridDomain::FullSphere;
        int powderPoints = 1;
        int powderGridSize = 4;
        std::string powderSymmetry = "c1";
        int powderGammaPoints = 1;
        double powderGammaOffset = 0.0;
        double explicitAlpha = 0.0;
        double explicitBeta = 0.0;
        double explicitGamma = 0.0;
        double explicitWeight = 1.0;

        bool IsPowder() const { return orientation != SSOrientationMode::Identity; }
        bool IsStaticSolve() const { return calculation != SSCalculation::TimeEvolution; }
    };

    bool ResolveSSExecutionPlan(const MSDParser::ObjectParser &_properties,
        SSExecutionPlan &_plan, std::string &_error);

    const char *ToString(SSCalculation _value);
    const char *ToString(SSPropagation _value);
    const char *ToString(SSObservableMode _value);
    const char *ToString(SSOrientationMode _value);
    const char *ToString(SSRelaxationModel _value);
}

#endif
