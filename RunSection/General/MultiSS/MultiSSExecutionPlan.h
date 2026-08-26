/////////////////////////////////////////////////////////////////////////
// MultiSSExecutionPlan (RunSection::General::MultiSS)
// ------------------
// Orthogonal calculation policy for TaskMultiSSGeneral.
//
// HIERARCHY CONTRACT
//   This layer owns *network execution policy* only.  It does not construct
//   spin operators (SpinAPI), local one-system generators (General/SS), or
//   output streams (TaskMultiSSGeneral).  Keeping these concerns separate is
//   intentional and mirrors the successful General/HS architecture.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_MultiSS_MultiSSExecutionPlan
#define MOD_RunSection_General_MultiSS_MultiSSExecutionPlan

#include <string>
#include "MSDParserfwd.h"
#include "PowderGrid.h"
#include "SSLiouvillianBuilder.h"

namespace RunSection::General::MultiSS
{
    enum class MultiSSCalculation { TimeEvolution, TimeIntegrated, SteadyState };
    enum class MultiSSPropagation { RK4, Exponential };
    enum class MultiSSObservableMode { Populations, States, Both };
    enum class MultiSSOrientationMode { Identity, Explicit, Powder2D, PowderSO3 };

    struct MultiSSExecutionPlan
    {
        MultiSSCalculation calculation = MultiSSCalculation::TimeEvolution;
        MultiSSPropagation propagation = MultiSSPropagation::RK4;
        MultiSSObservableMode observables = MultiSSObservableMode::Both;
        ::RunSection::General::SS::SSHamiltonianMode hamiltonianMode =
            ::RunSection::General::SS::SSHamiltonianMode::FixedFull;
        bool historicalNZ = false;

        double totalTime = 1000.0;
        double timeStep = 1.0;
        double solverResidualTolerance = 1.0e-8;
        double probabilityTolerance = 1.0e-9;
        bool diagnostics = true;
        bool transitionFluxes = false;

        MultiSSOrientationMode orientation = MultiSSOrientationMode::Identity;
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

        bool IsPowder() const { return orientation != MultiSSOrientationMode::Identity; }
        bool IsStaticSolve() const { return calculation != MultiSSCalculation::TimeEvolution; }
    };

    bool ResolveMultiSSExecutionPlan(const MSDParser::ObjectParser &_properties,
        MultiSSExecutionPlan &_plan, std::string &_error);

    const char *ToString(MultiSSCalculation _value);
    const char *ToString(MultiSSPropagation _value);
    const char *ToString(MultiSSObservableMode _value);
    const char *ToString(MultiSSOrientationMode _value);
}

#endif
