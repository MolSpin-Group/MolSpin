/////////////////////////////////////////////////////////////////////////
// SSExecutionPlan implementation.
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
        const bool modeSpecified=ReadString(p,{"hamiltonianmode","hamiltonian_mode","approximation"},value);
        if(value=="full"||value=="fixed"||value=="exact"||value=="nonsecular") plan.hamiltonianMode=SSHamiltonianMode::FixedFull;
        else if(value=="rotated_zyz"||value=="rotatedzyz") plan.hamiltonianMode=SSHamiltonianMode::RotatedFull;
        else if(value=="secular"||value=="highfield"||value=="high-field"||value=="rotated_sa"||value=="rotatedsa") plan.hamiltonianMode=SSHamiltonianMode::RotatedSecular;
        else {error="hamiltonian mode/approximation must be full, rotated_zyz, or secular/highfield";return false;}

        value="operators"; ReadString(p,{"relaxationmodel","relaxation_model"},value);
        if(value=="operators"||value=="operator"||value=="local") plan.relaxationModel=SSRelaxationModel::Operators;
        else if(value=="historical_nz"||value=="historical-nz"||value=="nakajimazwanzig"||value=="nz") plan.relaxationModel=SSRelaxationModel::HistoricalNZ;
        else {error="relaxationmodel must be operators or historical_nz/nakajimazwanzig";return false;}

        value="haberkorn"; ReadString(p,{"reactionoperators","reactionoperator","reaction_operator"},value);
        if(value=="haberkorn"||value=="hab") plan.reactionOperator=SpinAPI::ReactionOperatorType::Haberkorn;
        else if(value=="lindblad"||value=="gksl")
        {error="SSGeneral Lindblad reaction-operator parity is not yet qualified; use reactionoperators=haberkorn or a historical task";return false;}
        else {error="reactionoperators must be haberkorn (Lindblad is currently parity-gated)";return false;}

        std::string explicitOrientation;
        if(p.Get("powderorientation",explicitOrientation)||p.Get("orientation",explicitOrientation))
        {
            if(!ParseOrientation(explicitOrientation,plan.explicitAlpha,plan.explicitBeta,plan.explicitGamma,plan.explicitWeight))
            {error="powderorientation must contain alpha beta [gamma [weight]]";return false;}
            plan.orientation=SSOrientationMode::Explicit;
        }

        std::string grid; const bool gridSpecified=ReadString(p,{"powdergrid","powder_grid"},grid);
        const bool pointsSpecified=p.Get("powdersamplingpoints",plan.powderPoints);
        p.Get("powdergridsize",plan.powderGridSize); p.Get("powdersymmetry",plan.powderSymmetry);
        p.Get("powdergammapoints",plan.powderGammaPoints); p.Get("powdergamma",plan.powderGammaOffset);
        if(gridSpecified)
        {
            if(plan.orientation==SSOrientationMode::Explicit){error="explicit powderorientation cannot be combined with powdergrid";return false;}
            if(grid=="uniform"||grid=="golden"||grid=="fibonacci") plan.powderGridType=SpinAPI::PowderGridType::Uniform;
            else if(grid=="octant") plan.powderGridType=SpinAPI::PowderGridType::Octant;
            else if(grid=="sophe") plan.powderGridType=SpinAPI::PowderGridType::Sophe;
            else {error="powdergrid must be uniform, octant, or sophe";return false;}
            plan.orientation=plan.powderGammaPoints>1?SSOrientationMode::PowderSO3:SSOrientationMode::Powder2D;
        }
        else if(pointsSpecified&&plan.powderPoints>1)
            plan.orientation=plan.powderGammaPoints>1?SSOrientationMode::PowderSO3:SSOrientationMode::Powder2D;

        std::string domain;
        if(ReadString(p,{"powderdomain"},domain))
        {
            if(domain=="full"||domain=="sphere"||domain=="fullsphere") plan.powderDomain=SpinAPI::PowderGridDomain::FullSphere;
            else if(domain=="upper"||domain=="hemisphere"||domain=="upperhemisphere") plan.powderDomain=SpinAPI::PowderGridDomain::UpperHemisphere;
            else {error="powderdomain must be full or upper";return false;}
        }
        if(plan.orientation==SSOrientationMode::PowderSO3&&plan.powderGammaPoints<2){error="SO(3) powder sampling requires powdergammapoints >= 2";return false;}
        if((plan.orientation==SSOrientationMode::Powder2D||plan.orientation==SSOrientationMode::PowderSO3)&&
            plan.powderGridType!=SpinAPI::PowderGridType::Sophe&&plan.powderPoints<1)
        {error="generated powder grid requires positive powdersamplingpoints";return false;}

        // A full powder calculation must rotate the Hamiltonian even when the
        // user expressed the request simply as approximation=full.
        if(plan.orientation!=SSOrientationMode::Identity&&plan.orientation!=SSOrientationMode::Explicit &&
            plan.hamiltonianMode==SSHamiltonianMode::FixedFull)
            plan.hamiltonianMode=SSHamiltonianMode::RotatedFull;
        if(plan.orientation==SSOrientationMode::Explicit&&plan.hamiltonianMode==SSHamiltonianMode::FixedFull&&modeSpecified)
            plan.hamiltonianMode=SSHamiltonianMode::RotatedFull;
        return true;
    }

    const char *ToString(SSCalculation v){switch(v){case SSCalculation::TimeEvolution:return "timeevolution";case SSCalculation::TimeIntegrated:return "timeintegrated";case SSCalculation::SteadyState:return "steadystate";}return "unknown";}
    const char *ToString(SSPropagation v){return v==SSPropagation::Exponential?"exponential":"rk4";}
    const char *ToString(SSObservableMode v){switch(v){case SSObservableMode::States:return "states";case SSObservableMode::TransitionYields:return "transitionyields";case SSObservableMode::Both:return "both";}return "unknown";}
    const char *ToString(SSRelaxationModel v){return v==SSRelaxationModel::HistoricalNZ?"historical_nz":"operators";}
    const char *ToString(SSOrientationMode v){switch(v){case SSOrientationMode::Identity:return "identity";case SSOrientationMode::Explicit:return "explicit";case SSOrientationMode::Powder2D:return "powder2d";case SSOrientationMode::PowderSO3:return "powderso3";}return "unknown";}
}
