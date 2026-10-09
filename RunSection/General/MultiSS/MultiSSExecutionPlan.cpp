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
/////////////////////////////////////////////////////////////////////////
// DEVELOPER WORKFLOW / OWNERSHIP MAP
// ----------------------------------------------------------------------
// MultiSSGeneral input normalization and policy gate.
//
// What is done here:
//   - Parses calculation, local Hamiltonian approximation, relaxation model, powder sampling and propagator settings.
//   - Rejects incompatible static-solve, event and time-dependent-network combinations early.
//
// Connections to the General framework / SpinAPI:
//   - TaskMultiSSGeneral consumes this plan.
//   - Local one-manifold semantics are deliberately aligned with SSExecutionPlan because MultiSS reuses SS local builders.
//
// Why this ownership is used:
//   - MultiSS policy describes a kinetic network of Liouville manifolds; it must not silently inherit HS-only amplitude propagation semantics.
//
// TODO:
//   - Add ensemble-realization settings here only through the same shared abstraction used by HSGeneral and SSGeneral.
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
        p.Get("krylovdimension", plan.krylovDimension);
        p.Get("krylovtolerance", plan.krylovTolerance);
        p.Get("diagnostics", plan.diagnostics);
        p.Get("transitionfluxes", plan.transitionFluxes);
        if (!std::isfinite(plan.totalTime) || plan.totalTime < 0.0)
        { error="totaltime must be finite and non-negative"; return false; }
        if (!std::isfinite(plan.timeStep) || !(plan.timeStep > 0.0))
        { error="timestep must be finite and positive"; return false; }
        if (!std::isfinite(plan.solverResidualTolerance) || !(plan.solverResidualTolerance > 0.0))
        { error="solverresidualtolerance must be finite and positive"; return false; }
        if (plan.krylovDimension < 2)
        { error="krylovdimension must be at least two"; return false; }
        if (!std::isfinite(plan.krylovTolerance) || !(plan.krylovTolerance > 0.0))
        { error="krylovtolerance must be finite and positive"; return false; }

        value = "rk4";
        ReadString(p,{"propagationmethod","propagator"},value);
        if (value=="rk4" || value=="explicit") plan.propagation=MultiSSPropagation::RK4;
        else if (value=="exponential" || value=="exp" || value=="normal") plan.propagation=MultiSSPropagation::Exponential;
        else if (value=="krylov" || value=="arnoldi" || value=="expmv") plan.propagation=MultiSSPropagation::Krylov;
        else { error="propagationmethod must be rk4, exponential, or krylov"; return false; }

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
        const bool explicitSpecified =
            p.Get("powderorientation", explicitOrientation) || p.Get("orientation", explicitOrientation);
        if (explicitSpecified)
        {
            if (!ParseOrientation(explicitOrientation, plan.explicitAlpha, plan.explicitBeta,
                plan.explicitGamma, plan.explicitWeight))
            { error="powderorientation must contain alpha beta [gamma [weight]]"; return false; }
            if (!(plan.explicitWeight > 0.0))
            { error="explicit powderorientation weight must be positive"; return false; }
            plan.orientation=MultiSSOrientationMode::Explicit;
        }

        std::string grid;
        const bool gridSpecified = ReadString(p,{"powdergrid","powder_grid"},grid);
        const bool pointsSpecified = p.Get("powdersamplingpoints",plan.powderPoints);
        const bool gridSizeSpecified = p.Get("powdergridsize",plan.powderGridSize);
        const bool symmetrySpecified = p.Get("powdersymmetry",plan.powderSymmetry);
        const bool gammaPointsSpecified = p.Get("powdergammapoints",plan.powderGammaPoints);
        const bool gammaOffsetSpecified = p.Get("powdergamma",plan.powderGammaOffset);
        if (plan.powderGammaPoints < 1)
        { error="powdergammapoints must be at least one"; return false; }

        std::string domain;
        const bool domainSpecified = ReadString(p,{"powderdomain"},domain);
        if (explicitSpecified && (gridSpecified || pointsSpecified || gridSizeSpecified ||
            symmetrySpecified || domainSpecified || gammaOffsetSpecified || plan.powderGammaPoints > 1))
        { error="explicit powderorientation cannot be combined with generated powder-grid settings"; return false; }

        bool generatedPowderGrid = false;
        if (gridSpecified)
        {
            if (grid=="uniform" || grid=="golden" || grid=="fibonacci") plan.powderGridType=SpinAPI::PowderGridType::Uniform;
            else if (grid=="octant") plan.powderGridType=SpinAPI::PowderGridType::Octant;
            else if (grid=="sophe") plan.powderGridType=SpinAPI::PowderGridType::Sophe;
            else { error="powdergrid must be uniform, octant, or sophe"; return false; }
        }

        if (domainSpecified)
        {
            if (domain=="full" || domain=="sphere" || domain=="fullsphere") plan.powderDomain=SpinAPI::PowderGridDomain::FullSphere;
            else if (domain=="upper" || domain=="hemisphere" || domain=="upperhemisphere") plan.powderDomain=SpinAPI::PowderGridDomain::UpperHemisphere;
            else { error="powderdomain must be full or upper"; return false; }
        }

        if (!explicitSpecified)
        {
            switch (plan.powderGridType)
            {
            case SpinAPI::PowderGridType::Uniform:
                if (gridSizeSpecified || symmetrySpecified)
                { error="powdergridsize/powdersymmetry are only valid with powdergrid=sophe"; return false; }
                if (gridSpecified && !pointsSpecified)
                { error="powdergrid=uniform requires powdersamplingpoints"; return false; }
                if (pointsSpecified && plan.powderPoints < 1)
                { error="powdersamplingpoints must be at least one"; return false; }
                generatedPowderGrid=(gridSpecified&&pointsSpecified)||plan.powderPoints>1;
                break;
            case SpinAPI::PowderGridType::Sophe:
                if (pointsSpecified)
                { error="powdersamplingpoints is not used by powdergrid=sophe; use powdergridsize"; return false; }
                if (domainSpecified)
                { error="powderdomain does not apply to powdergrid=sophe; powdersymmetry defines its domain"; return false; }
                if (plan.powderGridSize < 1)
                { error="powdergridsize must be at least one for powdergrid=sophe"; return false; }
                {
                    SpinAPI::SopheGridParameters parameters;
                    if (!SpinAPI::GetSopheGridParameters(plan.powderSymmetry,parameters))
                    { error="invalid powdersymmetry for powdergrid=sophe"; return false; }
                }
                generatedPowderGrid=gridSpecified;
                break;
            case SpinAPI::PowderGridType::Octant:
                if (!gridSpecified)
                { error="powdergrid=octant must be selected explicitly"; return false; }
                if (!pointsSpecified || plan.powderPoints < 1)
                { error="powdergrid=octant requires positive powdersamplingpoints"; return false; }
                if (gridSizeSpecified || symmetrySpecified)
                { error="powdergridsize/powdersymmetry are only valid with powdergrid=sophe"; return false; }
                if (domainSpecified)
                { error="powdergrid=octant has a fixed symmetry-reduced domain"; return false; }
                generatedPowderGrid=true;
                break;
            }
            if (domainSpecified && !generatedPowderGrid)
            { error="powderdomain requires a generated powder grid"; return false; }
            if (plan.powderGammaPoints > 1 && !generatedPowderGrid)
            { error="powdergammapoints greater than one requires a generated theta/phi powder grid"; return false; }
            if (gammaOffsetSpecified && plan.powderGammaPoints < 2)
            { error="powdergamma requires powdergammapoints greater than one"; return false; }
            if (gammaPointsSpecified && plan.powderGammaPoints > 1 &&
                plan.powderGridType==SpinAPI::PowderGridType::Uniform && plan.powderPoints<=1)
            { error="uniform SO(3) powder sampling requires powdersamplingpoints greater than one"; return false; }
            if (plan.powderGridType==SpinAPI::PowderGridType::Uniform &&
                plan.powderGammaPoints>1 && !domainSpecified)
            {
                // A generic SO(3) average requires full-sphere theta/phi
                // sampling once the third Euler angle is resolved.
                plan.powderDomain=SpinAPI::PowderGridDomain::FullSphere;
            }
            if (plan.powderGammaPoints > 1) plan.orientation=MultiSSOrientationMode::PowderSO3;
            else if (generatedPowderGrid) plan.orientation=MultiSSOrientationMode::Powder2D;
        }

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
    {
        switch(v)
        {
        case MultiSSPropagation::RK4:return "rk4";
        case MultiSSPropagation::Exponential:return "exponential";
        case MultiSSPropagation::Krylov:return "krylov";
        }
        return "unknown";
    }
    const char *ToString(MultiSSObservableMode v)
    { switch(v){case MultiSSObservableMode::Populations:return "populations";case MultiSSObservableMode::States:return "states";case MultiSSObservableMode::Both:return "both";} return "unknown"; }
    const char *ToString(MultiSSOrientationMode v)
    { switch(v){case MultiSSOrientationMode::Identity:return "identity";case MultiSSOrientationMode::Explicit:return "explicit";case MultiSSOrientationMode::Powder2D:return "powder2d";case MultiSSOrientationMode::PowderSO3:return "powderso3";} return "unknown"; }
}
