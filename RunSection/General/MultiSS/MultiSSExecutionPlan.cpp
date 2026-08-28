/////////////////////////////////////////////////////////////////////////
// MultiSSExecutionPlan implementation (RunSection::General::MultiSS)
// ----------------------------------------------------------------------
// This file normalizes user input into orthogonal execution choices.  It does
// not build Hamiltonians, transfer maps, or propagate a state.  Keeping parsing
// policy separate from physics mirrors General/HS and makes unsupported model
// combinations fail explicitly instead of falling through to a legacy task.
//
// CALCULATION MODES ARE MATHEMATICALLY DISTINCT
//   timeevolution : d rho/dt = L(t) rho and explicit event handling.
//   timeintegrated/timeinf : X=int_0^inf rho(t)dt, so static homogeneous
//                            decay obeys L X = -rho(0).
//   steadystate : L rho_ss=0 with Tr rho_ss=1 for a closed trace-preserving
//                 represented network.
//
// A finite one-shot Gaussian optical pulse therefore belongs to timeevolution,
// not to an ordinary static steady-state solve.  Periodically repeated pulses
// will require a future cycle-map fixed point rather than L rho=0.
//
// HIERARCHY: execution policy -> General/MultiSS engines -> TaskMultiSSGeneral.
// No RunSection/Tasks legacy class is selected from this plan.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "MultiSSExecutionPlan.h"
#include "ObjectParser.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>

namespace RunSection::General::MultiSS
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
            for (const char *key : keys)
                if (p.Get(key, value)) { value = Lower(value); return true; }
            return false;
        }
        bool ParseOrientation(const std::string &text, double &a, double &b,
            double &g, double &w)
        {
            std::string s = text;
            for (char &c : s) if (c==',' || c=='[' || c==']' || c=='(' || c==')') c=' ';
            std::istringstream in(s);
            if (!(in >> a >> b)) return false;
            if (!(in >> g)) g = 0.0;
            if (!(in >> w)) w = 1.0;
            return std::isfinite(a)&&std::isfinite(b)&&std::isfinite(g)&&std::isfinite(w);
        }
    }

    bool ResolveMultiSSExecutionPlan(const MSDParser::ObjectParser &p,
        MultiSSExecutionPlan &plan, std::string &error)
    {
        plan = MultiSSExecutionPlan(); error.clear();
        std::string value = "timeevolution";
        ReadString(p, {"calculation","calculationmode","calculation_mode"}, value);
        if (value=="timeevolution" || value=="timeevo" || value=="time-evolution")
            plan.calculation = MultiSSCalculation::TimeEvolution;
        else if (value=="timeintegrated" || value=="time-integrated" || value=="timeinf" || value=="integrated")
            plan.calculation = MultiSSCalculation::TimeIntegrated;
        else if (value=="steadystate" || value=="steady-state" || value=="stationary")
            plan.calculation = MultiSSCalculation::SteadyState;
        else { error="calculation must be timeevolution, timeintegrated/timeinf, or steadystate"; return false; }

        p.Get("totaltime", plan.totalTime); p.Get("timestep", plan.timeStep);
        p.Get("solverresidualtolerance", plan.solverResidualTolerance);
        p.Get("diagnostics", plan.diagnostics);
        p.Get("transitionfluxes", plan.transitionFluxes);
        if (!std::isfinite(plan.totalTime) || plan.totalTime < 0.0)
        { error="totaltime must be finite and non-negative"; return false; }
        if (!std::isfinite(plan.timeStep) || !(plan.timeStep > 0.0))
        { error="timestep must be finite and positive"; return false; }
        if (!std::isfinite(plan.solverResidualTolerance) || !(plan.solverResidualTolerance > 0.0))
        { error="solverresidualtolerance must be finite and positive"; return false; }

        value = "rk4";
        ReadString(p,{"propagationmethod","propagator"},value);
        if (value=="rk4" || value=="explicit") plan.propagation=MultiSSPropagation::RK4;
        else if (value=="exponential" || value=="exp" || value=="normal") plan.propagation=MultiSSPropagation::Exponential;
        else { error="propagationmethod must be rk4 or exponential"; return false; }

        value = "both";
        ReadString(p,{"observables","observablemode"},value);
        if (value=="populations" || value=="population" || value=="manifolds") plan.observables=MultiSSObservableMode::Populations;
        else if (value=="states" || value=="statepopulations") plan.observables=MultiSSObservableMode::States;
        else if (value=="both" || value=="all") plan.observables=MultiSSObservableMode::Both;
        else { error="observables must be populations, states, or both"; return false; }

        value = "full";
        ReadString(p,{"hamiltonianmode","hamiltonian_mode"},value);
        if (value=="full" || value=="fixed" || value=="legacy")
            plan.hamiltonianMode=::RunSection::General::SS::SSHamiltonianMode::FixedFull;
        else if (value=="rotated_zyz" || value=="rotatedzyz" || value=="exact" || value=="nonsecular")
            plan.hamiltonianMode=::RunSection::General::SS::SSHamiltonianMode::RotatedFull;
        else if (value=="rotated_sa" || value=="rotatedsa" || value=="secular" || value=="highfield")
            plan.hamiltonianMode=::RunSection::General::SS::SSHamiltonianMode::RotatedSecular;
        else { error="hamiltonianmode must be full, rotated_zyz, or rotated_sa"; return false; }

        value = "operators";
        ReadString(p,{"relaxationmodel","relaxation_model"},value);
        if (value=="operators" || value=="operator" || value=="local")
            plan.relaxationModel=::RunSection::General::SS::SSRelaxationModel::Operators;
        else if (value=="nakajima_zwanzig" || value=="nakajima-zwanzig" || value=="nakajimazwanzig" || value=="nz")
            plan.relaxationModel=::RunSection::General::SS::SSRelaxationModel::NakajimaZwanzig;
        else if (value=="redfield")
            plan.relaxationModel=::RunSection::General::SS::SSRelaxationModel::Redfield;
        else { error="relaxationmodel must be operators, nakajima_zwanzig, or redfield"; return false; }

        std::string explicitOrientation;
        if (p.Get("powderorientation", explicitOrientation) || p.Get("orientation", explicitOrientation))
        {
            if (!ParseOrientation(explicitOrientation, plan.explicitAlpha, plan.explicitBeta,
                plan.explicitGamma, plan.explicitWeight))
            { error="powderorientation must contain alpha beta [gamma [weight]]"; return false; }
            plan.orientation=MultiSSOrientationMode::Explicit;
        }

        std::string grid;
        const bool gridSpecified = ReadString(p,{"powdergrid","powder_grid"},grid);
        const bool pointsSpecified = p.Get("powdersamplingpoints",plan.powderPoints);
        p.Get("powdergridsize",plan.powderGridSize);
        p.Get("powdersymmetry",plan.powderSymmetry);
        p.Get("powdergammapoints",plan.powderGammaPoints);
        p.Get("powdergamma",plan.powderGammaOffset);
        if (gridSpecified)
        {
            if (plan.orientation==MultiSSOrientationMode::Explicit)
            { error="explicit powderorientation cannot be combined with powdergrid"; return false; }
            if (grid=="uniform" || grid=="golden" || grid=="fibonacci") plan.powderGridType=SpinAPI::PowderGridType::Uniform;
            else if (grid=="octant") plan.powderGridType=SpinAPI::PowderGridType::Octant;
            else if (grid=="sophe") plan.powderGridType=SpinAPI::PowderGridType::Sophe;
            else { error="powdergrid must be uniform, octant, or sophe"; return false; }
            plan.orientation = plan.powderGammaPoints>1 ? MultiSSOrientationMode::PowderSO3 : MultiSSOrientationMode::Powder2D;
        }
        else if (pointsSpecified && plan.powderPoints>1)
            plan.orientation = plan.powderGammaPoints>1 ? MultiSSOrientationMode::PowderSO3 : MultiSSOrientationMode::Powder2D;

        std::string domain;
        if (ReadString(p,{"powderdomain"},domain))
        {
            if (domain=="full" || domain=="sphere" || domain=="fullsphere") plan.powderDomain=SpinAPI::PowderGridDomain::FullSphere;
            else if (domain=="upper" || domain=="hemisphere" || domain=="upperhemisphere") plan.powderDomain=SpinAPI::PowderGridDomain::UpperHemisphere;
            else { error="powderdomain must be full or upper"; return false; }
        }
        if (plan.orientation==MultiSSOrientationMode::PowderSO3 && plan.powderGammaPoints<2)
        { error="SO(3) powder sampling requires powdergammapoints >= 2"; return false; }
        if ((plan.orientation==MultiSSOrientationMode::Powder2D || plan.orientation==MultiSSOrientationMode::PowderSO3) &&
            plan.powderGridType!=SpinAPI::PowderGridType::Sophe && plan.powderPoints<1)
        { error="generated powder grid requires positive powdersamplingpoints"; return false; }

        // Every manifold of a molecular network follows the same crystallite.
        // A full Hamiltonian therefore uses the rotated full construction for
        // explicit orientations and generated powder grids alike.
        if (plan.orientation!=MultiSSOrientationMode::Identity &&
            plan.hamiltonianMode==::RunSection::General::SS::SSHamiltonianMode::FixedFull)
            plan.hamiltonianMode=::RunSection::General::SS::SSHamiltonianMode::RotatedFull;

        // A static linear solve has one fixed generator.  Finite Gaussian/
        // trajectory channels and instantaneous events are therefore rejected
        // later after the network has been compiled, where those facts are known.
        return true;
    }

    const char *ToString(MultiSSCalculation v)
    { switch(v){case MultiSSCalculation::TimeEvolution:return "timeevolution";case MultiSSCalculation::TimeIntegrated:return "timeintegrated";case MultiSSCalculation::SteadyState:return "steadystate";} return "unknown"; }
    const char *ToString(MultiSSPropagation v)
    { return v==MultiSSPropagation::RK4?"rk4":"exponential"; }
    const char *ToString(MultiSSObservableMode v)
    { switch(v){case MultiSSObservableMode::Populations:return "populations";case MultiSSObservableMode::States:return "states";case MultiSSObservableMode::Both:return "both";} return "unknown"; }
    const char *ToString(MultiSSOrientationMode v)
    { switch(v){case MultiSSOrientationMode::Identity:return "identity";case MultiSSOrientationMode::Explicit:return "explicit";case MultiSSOrientationMode::Powder2D:return "powder2d";case MultiSSOrientationMode::PowderSO3:return "powderso3";} return "unknown"; }
}
