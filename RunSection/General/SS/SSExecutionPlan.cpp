/////////////////////////////////////////////////////////////////////////
// SSExecutionPlan implementation (RunSection::General::SS)
// ------------------
// Resolves user input into an explicit, validated superspace calculation plan.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// DEVELOPER WORKFLOW / OWNERSHIP MAP
// ----------------------------------------------------------------------
// SSGeneral input normalization and policy gate.
//
// What is done here:
//   - Parses calculation mode, propagation, observables, orientation, Hamiltonian approximation and relaxation model.
//   - Rejects unsupported time-dependent or reaction-model combinations before Liouvillian construction.
//
// Connections to the General framework / SpinAPI:
//   - TaskSSGeneral consumes the plan; MultiSSExecutionPlan intentionally mirrors local SS vocabulary.
//   - SpinAPI provides reaction/operator and powder-grid enums.
//
// Why this ownership is used:
//   - SS policy describes one Liouville-space manifold; inter-system kinetics belong to MultiSSGeneral.
//
// TODO:
//   - Future strain/SW settings should be shared with HS/MultiSS through one realization abstraction, not added as SS-only keywords.
/////////////////////////////////////////////////////////////////////////

#include "SSExecutionPlan.h"
#include "ObjectParser.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>

namespace RunSection::General::SS
{
    namespace
    {
        std::string Lower(std::string value)
        {
            std::transform(value.begin(), value.end(), value.begin(),
                [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
            return value;
        }
        bool ReadString(const MSDParser::ObjectParser &p,
            std::initializer_list<const char*> keys, std::string &value)
        {
            for(const char *key:keys) if(p.Get(key,value)){value=Lower(value);return true;}
            return false;
        }
        bool ParseOrientation(const std::string &text,double &a,double &b,double &g,double &w)
        {
            std::string s=text; for(char &c:s) if(c==','||c=='['||c==']'||c=='('||c==')') c=' ';
            std::istringstream in(s);
            if(!(in>>a>>b))
                return false;
            if(!(in>>g))
                g=0.0;
            if(!(in>>w))
                w=1.0;
            return std::isfinite(a)&&std::isfinite(b)&&std::isfinite(g)&&std::isfinite(w);
        }
    }

    bool ResolveSSExecutionPlan(const MSDParser::ObjectParser &p,SSExecutionPlan &plan,std::string &error)
    {
        plan=SSExecutionPlan(); error.clear(); std::string value="timeintegrated";
        ReadString(p,{"calculation","calculationmode","calculation_mode","method"},value);
        if(value=="timeevolution"||value=="timeevo"||value=="time-evolution") plan.calculation=SSCalculation::TimeEvolution;
        else if(value=="timeintegrated"||value=="time-integrated"||value=="timeinf"||value=="integrated"||value=="static") plan.calculation=SSCalculation::TimeIntegrated;
        else if(value=="steadystate"||value=="steady-state"||value=="stationary") plan.calculation=SSCalculation::SteadyState;
        else {error="calculation must be timeevolution, timeintegrated/timeinf, or steadystate";return false;}

        p.Get("totaltime",plan.totalTime); p.Get("timestep",plan.timeStep);
        p.Get("solverresidualtolerance",plan.solverResidualTolerance); p.Get("diagnostics",plan.diagnostics);
        if(!std::isfinite(plan.totalTime)||plan.totalTime<0.0){error="totaltime must be finite and non-negative";return false;}
        if(!std::isfinite(plan.timeStep)||!(plan.timeStep>0.0)){error="timestep must be finite and positive";return false;}
        if(!std::isfinite(plan.solverResidualTolerance)||!(plan.solverResidualTolerance>0.0)){error="solverresidualtolerance must be finite and positive";return false;}

        value="exponential"; ReadString(p,{"propagationmethod","propagator"},value);
        if(value=="exponential"||value=="exp"||value=="normal"||value=="auto") plan.propagation=SSPropagation::Exponential;
        else if(value=="rk4"||value=="explicit") plan.propagation=SSPropagation::RK4;
        else {error="propagationmethod must be exponential or rk4";return false;}

        value=plan.calculation==SSCalculation::TimeIntegrated?"states":"states";
        ReadString(p,{"observables","observablemode"},value);
        if(value=="states"||value=="statepopulations"||value=="populations") plan.observables=SSObservableMode::States;
        else if(value=="transitions"||value=="transitionyields"||value=="products"||value=="yields") plan.observables=SSObservableMode::TransitionYields;
        else if(value=="both"||value=="all") plan.observables=SSObservableMode::Both;
        else {error="observables must be states, transitionyields, or both";return false;}

        value="full";
        ReadString(p,{"hamiltonianmode","hamiltonian_mode","approximation"},value);
        if(value=="full"||value=="fixed"||value=="exact"||value=="nonsecular") plan.hamiltonianMode=SSHamiltonianMode::FixedFull;
        else if(value=="rotated_zyz"||value=="rotatedzyz") plan.hamiltonianMode=SSHamiltonianMode::RotatedFull;
        else if(value=="secular"||value=="highfield"||value=="high-field"||value=="rotated_sa"||value=="rotatedsa") plan.hamiltonianMode=SSHamiltonianMode::RotatedSecular;
        else {error="hamiltonian mode/approximation must be full, rotated_zyz, or secular/highfield";return false;}

        value="operators"; ReadString(p,{"relaxationmodel","relaxation_model"},value);
        if(value=="operators"||value=="operator"||value=="local") plan.relaxationModel=SSRelaxationModel::Operators;
        else if(value=="nakajima_zwanzig"||value=="nakajima-zwanzig"||value=="nakajimazwanzig"||value=="nz") plan.relaxationModel=SSRelaxationModel::NakajimaZwanzig;
        else if(value=="redfield") plan.relaxationModel=SSRelaxationModel::Redfield;
        else {error="relaxationmodel must be operators, nakajima_zwanzig, or redfield";return false;}

        value="haberkorn"; ReadString(p,{"reactionoperators","reactionoperator","reaction_operator"},value);
        if(value=="haberkorn"||value=="hab") plan.reactionOperator=SpinAPI::ReactionOperatorType::Haberkorn;
        else if(value=="lindblad"||value=="gksl")
        {error="SSGeneral Lindblad reaction-operator parity is not yet qualified; use reactionoperators=haberkorn or a dedicated task";return false;}
        else {error="reactionoperators must be haberkorn (Lindblad is currently parity-gated)";return false;}

        std::string explicitOrientation;
        const bool explicitSpecified =
            p.Get("powderorientation",explicitOrientation)||p.Get("orientation",explicitOrientation);
        if(explicitSpecified)
        {
            if(!ParseOrientation(explicitOrientation,plan.explicitAlpha,plan.explicitBeta,plan.explicitGamma,plan.explicitWeight))
            {error="powderorientation must contain alpha beta [gamma [weight]]";return false;}
            if(!(plan.explicitWeight>0.0))
            {error="explicit powderorientation weight must be positive";return false;}
            plan.orientation=SSOrientationMode::Explicit;
        }

        std::string grid; const bool gridSpecified=ReadString(p,{"powdergrid","powder_grid"},grid);
        const bool pointsSpecified=p.Get("powdersamplingpoints",plan.powderPoints);
        const bool gridSizeSpecified=p.Get("powdergridsize",plan.powderGridSize);
        const bool symmetrySpecified=p.Get("powdersymmetry",plan.powderSymmetry);
        const bool gammaPointsSpecified=p.Get("powdergammapoints",plan.powderGammaPoints);
        const bool gammaOffsetSpecified=p.Get("powdergamma",plan.powderGammaOffset);
        if(plan.powderGammaPoints<1){error="powdergammapoints must be at least one";return false;}

        std::string domain;
        const bool domainSpecified=ReadString(p,{"powderdomain"},domain);
        if(explicitSpecified&&(gridSpecified||pointsSpecified||gridSizeSpecified||symmetrySpecified||
            domainSpecified||gammaOffsetSpecified||plan.powderGammaPoints>1))
        {error="explicit powderorientation cannot be combined with generated powder-grid settings";return false;}

        bool generatedPowderGrid=false;
        if(gridSpecified)
        {
            if(grid=="uniform"||grid=="golden"||grid=="fibonacci") plan.powderGridType=SpinAPI::PowderGridType::Uniform;
            else if(grid=="octant") plan.powderGridType=SpinAPI::PowderGridType::Octant;
            else if(grid=="sophe") plan.powderGridType=SpinAPI::PowderGridType::Sophe;
            else {error="powdergrid must be uniform, octant, or sophe";return false;}
        }

        if(domainSpecified)
        {
            if(domain=="full"||domain=="sphere"||domain=="fullsphere") plan.powderDomain=SpinAPI::PowderGridDomain::FullSphere;
            else if(domain=="upper"||domain=="hemisphere"||domain=="upperhemisphere") plan.powderDomain=SpinAPI::PowderGridDomain::UpperHemisphere;
            else {error="powderdomain must be full or upper";return false;}
        }

        if(!explicitSpecified)
        {
            switch(plan.powderGridType)
            {
            case SpinAPI::PowderGridType::Uniform:
                if(gridSizeSpecified||symmetrySpecified)
                {error="powdergridsize/powdersymmetry are only valid with powdergrid=sophe";return false;}
                if(gridSpecified&&!pointsSpecified)
                {error="powdergrid=uniform requires powdersamplingpoints";return false;}
                if(pointsSpecified&&plan.powderPoints<1)
                {error="powdersamplingpoints must be at least one";return false;}
                generatedPowderGrid=(gridSpecified&&pointsSpecified)||plan.powderPoints>1;
                break;
            case SpinAPI::PowderGridType::Sophe:
                if(pointsSpecified)
                {error="powdersamplingpoints is not used by powdergrid=sophe; use powdergridsize";return false;}
                if(domainSpecified)
                {error="powderdomain does not apply to powdergrid=sophe; powdersymmetry defines its domain";return false;}
                if(plan.powderGridSize<1)
                {error="powdergridsize must be at least one for powdergrid=sophe";return false;}
                {
                    SpinAPI::SopheGridParameters parameters;
                    if(!SpinAPI::GetSopheGridParameters(plan.powderSymmetry,parameters))
                    {error="invalid powdersymmetry for powdergrid=sophe";return false;}
                }
                generatedPowderGrid=gridSpecified;
                break;
            case SpinAPI::PowderGridType::Octant:
                if(!gridSpecified)
                {error="powdergrid=octant must be selected explicitly";return false;}
                if(!pointsSpecified||plan.powderPoints<1)
                {error="powdergrid=octant requires positive powdersamplingpoints";return false;}
                if(gridSizeSpecified||symmetrySpecified)
                {error="powdergridsize/powdersymmetry are only valid with powdergrid=sophe";return false;}
                if(domainSpecified)
                {error="powdergrid=octant has a fixed symmetry-reduced domain";return false;}
                generatedPowderGrid=true;
                break;
            }
            if(domainSpecified&&!generatedPowderGrid)
            {error="powderdomain requires a generated powder grid";return false;}
            if(plan.powderGammaPoints>1&&!generatedPowderGrid)
            {error="powdergammapoints greater than one requires a generated theta/phi powder grid";return false;}
            if(gammaOffsetSpecified&&plan.powderGammaPoints<2)
            {error="powdergamma requires powdergammapoints greater than one";return false;}
            if(gammaPointsSpecified&&plan.powderGammaPoints>1&&
                plan.powderGridType==SpinAPI::PowderGridType::Uniform&&plan.powderPoints<=1)
            {error="uniform SO(3) powder sampling requires powdersamplingpoints greater than one";return false;}
            if(plan.powderGridType==SpinAPI::PowderGridType::Uniform&&
                plan.powderGammaPoints>1&&!domainSpecified)
            {
                // Sampling the third Euler angle removes the axial reduction.
                // Unless the user explicitly supplies another symmetry domain,
                // theta/phi must cover the full sphere for an SO(3) average.
                plan.powderDomain=SpinAPI::PowderGridDomain::FullSphere;
            }
            if(plan.powderGammaPoints>1) plan.orientation=SSOrientationMode::PowderSO3;
            else if(generatedPowderGrid) plan.orientation=SSOrientationMode::Powder2D;
        }

        // Any non-identity crystallite orientation must rotate a full
        // Hamiltonian. This includes an explicit one-point orientation.
        if(plan.orientation!=SSOrientationMode::Identity&&
            plan.hamiltonianMode==SSHamiltonianMode::FixedFull)
            plan.hamiltonianMode=SSHamiltonianMode::RotatedFull;
        return true;
    }

    const char *ToString(SSCalculation v){switch(v){case SSCalculation::TimeEvolution:return "timeevolution";case SSCalculation::TimeIntegrated:return "timeintegrated";case SSCalculation::SteadyState:return "steadystate";}return "unknown";}
    const char *ToString(SSPropagation v){return v==SSPropagation::Exponential?"exponential":"rk4";}
    const char *ToString(SSObservableMode v){switch(v){case SSObservableMode::States:return "states";case SSObservableMode::TransitionYields:return "transitionyields";case SSObservableMode::Both:return "both";}return "unknown";}
    const char *ToString(SSRelaxationModel v){switch(v){case SSRelaxationModel::Operators:return "operators";case SSRelaxationModel::NakajimaZwanzig:return "nakajima_zwanzig";case SSRelaxationModel::Redfield:return "redfield";}return "unknown";}
    const char *ToString(SSOrientationMode v){switch(v){case SSOrientationMode::Identity:return "identity";case SSOrientationMode::Explicit:return "explicit";case SSOrientationMode::Powder2D:return "powder2d";case SSOrientationMode::PowderSO3:return "powderso3";}return "unknown";}
}
