/////////////////////////////////////////////////////////////////////////
// MultiSSPropagator implementation (RunSection::General::MultiSS)
// ----------------------------------------------------------------------
// CONTINUOUS TIME EVOLUTION
//   For finite/time-dependent kinetic networks this layer solves
//       d rho/dt = L(t) rho
//   using RK4.  Matrix-exponential stepping is allowed only when the compiled
//   generator is genuinely time independent and there are no discontinuous
//   events.  Event-aware stepping shortens h so that an integration interval
//   never straddles a discontinuity.
//
// INSTANTANEOUS EVENTS
//   Event times are immutable network metadata; MultiSSEventController records
//   whether an event was consumed and applies the SpinAPI::QuantumMap exactly
//   once.  This supports the instantaneous pump/push limit used in
//       DOI: 10.1126/science.abl4254
//
// TIME-INTEGRATED VS STEADY STATE
//   These must not be conflated:
//       X = integral_0^inf rho(t) dt  =>  L X = -rho(0)
//   for a stable static homogeneous decay, whereas
//       L rho_ss = 0, Tr rho_ss = 1
//   defines a normalized stationary state of a closed trace-preserving network.
//   Both linear solves carry explicit residual checks.  A terminal sink is
//   incompatible with the normalized steady-state solve unless its product
//   manifold is explicitly represented.
//
// HIERARCHY
//   This file operates on a compiled MultiSSNetwork.  It contains no Hamiltonian
//   construction, no TransferChannel parsing, and no legacy-task dispatch.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "MultiSSPropagator.h"
#include "MultiSSEventController.h"
#include "SpinSpace.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace RunSection::General::MultiSS
{
    namespace
    {
        bool ApplyEventsAt(double time,MultiSSNetwork &network,MultiSSEventController &events,
            arma::cx_vec &state,std::string &error)
        { return events.ApplyAt(time,network,state,error,1.0e-11*std::max(1.0,std::abs(time))); }

        // The controller intentionally owns application state but event times are
        // immutable network metadata, so the propagator can find the next future
        // boundary directly and ApplyAt decides whether it was already consumed.
        double NextBoundary(const MultiSSNetwork &network,double time)
        {
            double next=std::numeric_limits<double>::infinity();
            const double eps=1.0e-12*std::max(1.0,std::abs(time));
            for(const auto&e:network.events)
                if(e.physical.EventTime()>time+eps) next=std::min(next,e.physical.EventTime());
            return next;
        }

        arma::cx_vec RK4Step(const MultiSSNetwork &network,const arma::cx_vec &x,double t,double h)
        {
            const arma::cx_vec k1=network.Generator(t)*x;
            const arma::cx_vec k2=network.Generator(t+0.5*h)*(x+0.5*h*k1);
            const arma::cx_vec k3=network.Generator(t+0.5*h)*(x+0.5*h*k2);
            const arma::cx_vec k4=network.Generator(t+h)*(x+h*k3);
            return x+(h/6.0)*(k1+2.0*k2+2.0*k3+k4);
        }

        // Exponential-midpoint propagation. Constant L gives exp(hL)x;
        // time-dependent L(t) uses the second-order midpoint/Magnus step.
        bool KrylovStep(const MultiSSExecutionPlan &plan,const MultiSSNetwork &network,
            const arma::cx_vec &x,double t,double h,arma::cx_vec &out,
            std::string &error,int depth=0)
        {
            if(depth>20)
            {error="Krylov MultiSS propagation exceeded the subdivision limit; reduce timestep or increase krylovdimension";return false;}
            if(network.systems.globalDimension>static_cast<arma::uword>(std::numeric_limits<int>::max()))
            {error="Krylov MultiSS propagation exceeds the supported integer dimension";return false;}

            const arma::sp_cx_mat L=network.Generator(t+0.5*h);
            const int n=static_cast<int>(network.systems.globalDimension);
            const int m=std::min(plan.krylovDimension,n);
            SpinAPI::SpinSpace helper;
            const auto step=helper.KrylovExpmGeneral(L,x,arma::cx_double(h,0.0),m,n);
            if(!step.result.is_finite() || !std::isfinite(step.error_estimate))
            {error="Krylov MultiSS propagation produced a non-finite result/error estimate";return false;}
            const double scale=std::max({1.0,arma::norm(x,2),arma::norm(step.result,2)});
            if(step.error_estimate<=plan.krylovTolerance*scale)
            {out=step.result;return true;}

            arma::cx_vec mid;
            if(!KrylovStep(plan,network,x,t,0.5*h,mid,error,depth+1))return false;
            return KrylovStep(plan,network,mid,t+0.5*h,0.5*h,out,error,depth+1);
        }
    }

    bool MultiSSPropagator::Propagate(const MultiSSExecutionPlan &plan,MultiSSNetwork &network,
        MultiSSTrajectory &trajectory,std::string &error)
    {
        trajectory=MultiSSTrajectory();error.clear();
        if(plan.calculation!=MultiSSCalculation::TimeEvolution)
        {error="MultiSSPropagator::Propagate requires calculation=timeevolution";return false;}
        if(plan.propagation==MultiSSPropagation::Exponential && !network.ContinuousGeneratorIsTimeIndependent())
        {error="exponential MultiSS propagation requires time-independent continuous rates; use krylov or rk4 for finite optical profiles";return false;}

        arma::cx_vec state=network.systems.initialState;
        MultiSSEventController events(network);
        if(!ApplyEventsAt(0.0,network,events,state,error))return false;
        trajectory.times.push_back(0.0);trajectory.states.push_back(state);
        double t=0.0;
        while(t<plan.totalTime-1.0e-14*std::max(1.0,plan.totalTime))
        {
            double h=std::min(plan.timeStep,plan.totalTime-t);
            const double boundary=NextBoundary(network,t);
            if(std::isfinite(boundary) && boundary<t+h-1.0e-13*std::max(1.0,boundary)) h=boundary-t;
            if(!(h>0.0)){error="MultiSS propagation encountered a non-positive time step";return false;}

            if(plan.propagation==MultiSSPropagation::RK4)
                state=RK4Step(network,state,t,h);
            else if(plan.propagation==MultiSSPropagation::Exponential)
                state=arma::expmat(arma::cx_mat(network.Generator(t))*h)*state;
            else
            {
                arma::cx_vec next;
                if(!KrylovStep(plan,network,state,t,h,next,error))return false;
                state=std::move(next);
            }
            if(!state.is_finite()){error="MultiSS propagation produced a non-finite state";return false;}
            t+=h;

            // Avoid a roundoff-sized final propagation interval.  Repeated
            // addition of timestep can leave t microscopically below
            // totalTime even when the physical endpoint has been reached.
            const double finalTolerance =
                1.0e-12*std::max(1.0,std::abs(plan.totalTime));
            if(std::abs(t-plan.totalTime)<=finalTolerance)
                t=plan.totalTime;

            // Snap to a scheduled event when roundoff is the only difference.
            for(const auto&e:network.events)
                if(std::abs(t-e.physical.EventTime())<1.0e-11*std::max(1.0,std::abs(t)))t=e.physical.EventTime();
            if(!ApplyEventsAt(t,network,events,state,error))return false;
            trajectory.times.push_back(t);trajectory.states.push_back(state);
        }
        return true;
    }

    bool MultiSSPropagator::SolveTimeIntegrated(const MultiSSExecutionPlan &plan,
        const MultiSSNetwork &network,arma::cx_vec &integrated,std::string &error)
    {
        error.clear();integrated.reset();
        if(!network.IsTimeIndependent())
        {error="timeintegrated solve requires one static homogeneous generator";return false;}
        const arma::sp_cx_mat L=network.Generator(0.0);
        const arma::cx_vec rhs=-network.systems.initialState;
        bool ok=false;
        try{ok=arma::spsolve(integrated,L,rhs,"superlu");}catch(...){ok=false;}
        if(!ok) ok=arma::solve(integrated,arma::cx_mat(L),rhs,arma::solve_opts::no_approx);
        if(!ok||!integrated.is_finite())
        {error="failed to solve L X=-rho0; the network may contain a non-decaying subspace";return false;}
        const double residual=arma::norm(L*integrated-rhs,2)/std::max(1.0,arma::norm(rhs,2));
        if(!std::isfinite(residual)||residual>plan.solverResidualTolerance)
        {error="timeintegrated solve failed residual tolerance";return false;}
        return true;
    }

    bool MultiSSPropagator::SolveSteadyState(const MultiSSExecutionPlan &plan,
        const MultiSSNetwork &network,arma::cx_vec &steady,std::string &error)
    {
        error.clear();steady.reset();
        if(!network.IsTimeIndependent())
        {error="steadystate requires a static generator";return false;}
        double traceResidual=0.0;
        if(!network.IsTracePreserving(plan.solverResidualTolerance,&traceResidual))
        {error="steadystate requires a closed trace-preserving represented network; terminal sinks require an explicit product manifold";return false;}

        const arma::cx_mat L(network.Generator(0.0));
        arma::cx_mat A=L;
        arma::cx_vec b(A.n_rows,arma::fill::zeros);
        // Replace one redundant stationarity equation with Tr(rho)=1.  This is
        // a constrained null-space solve, not the L X=-rho0 time-integral solve.
        A.row(A.n_rows-1)=network.systems.traceFunctional.t();
        b(A.n_rows-1)=1.0;
        bool ok=arma::solve(steady,A,b,arma::solve_opts::no_approx);
        if(!ok||!steady.is_finite()){error="failed to solve normalized MultiSS steady state";return false;}
        const double r=arma::norm(L*steady,2);
        const arma::cx_double tr=arma::cdot(network.systems.traceFunctional,steady);
        if(r>plan.solverResidualTolerance*std::max(1.0,arma::norm(L,"fro")) ||
            std::abs(tr-arma::cx_double(1.0,0.0))>10.0*plan.solverResidualTolerance)
        {error="MultiSS steady-state solution failed stationarity/normalization residual";return false;}
        return true;
    }
}
