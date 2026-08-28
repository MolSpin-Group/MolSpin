/////////////////////////////////////////////////////////////////////////
// SSPropagator implementation (RunSection::General::SS)
// ------------------
// Propagates or solves a time-independent superspace generator.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "SSPropagator.h"
#include <algorithm>
#include <cmath>
#include <limits>
namespace RunSection::General::SS
{
    namespace
    {
        arma::cx_vec RK4(const arma::sp_cx_mat &L,const arma::cx_vec &x,double h)
        {
            // Materialize every RK stage. Armadillo expression templates may
            // otherwise retain references to temporary stage expressions, which
            // is unsafe once the statement that created the temporary ends.
            const arma::cx_vec k1=L*x;
            const arma::cx_vec k2=L*(x+0.5*h*k1);
            const arma::cx_vec k3=L*(x+0.5*h*k2);
            const arma::cx_vec k4=L*(x+h*k3);
            return x+(h/6.0)*(k1+2.0*k2+2.0*k3+k4);
        }
    }
    bool SSPropagator::Propagate(const SSExecutionPlan &plan,const SSPreparedCalculation &prepared,SSTrajectory &trajectory,std::string &error)
    {
        trajectory=SSTrajectory();error.clear();if(plan.calculation!=SSCalculation::TimeEvolution){error="SSPropagator::Propagate requires timeevolution";return false;}
        arma::cx_vec state=prepared.initialState;trajectory.times.push_back(0.0);trajectory.states.push_back(state);
        if(plan.totalTime==0.0)return true;

        // Derive output times from integer indices.  Repeatedly adding dt can
        // leave t a few ulps below an exactly divisible total time and used to
        // produce a duplicate, almost-zero final propagation step.
        const double ratio=plan.totalTime/plan.timeStep;
        if(!std::isfinite(ratio) || ratio>static_cast<double>(std::numeric_limits<size_t>::max()))
        {error="SSGeneral propagation requires too many time steps";return false;}
        const double nearest=std::round(ratio);
        const double ratioTolerance=64.0*std::numeric_limits<double>::epsilon()*
            std::max(1.0,std::abs(ratio));
        const size_t steps=static_cast<size_t>(nearest>=1.0 && std::abs(ratio-nearest)<=ratioTolerance?
            nearest:std::ceil(ratio));
        const double timeTolerance=64.0*std::numeric_limits<double>::epsilon()*
            std::max({1.0,plan.totalTime,plan.timeStep});

        // SSGeneral's generator is static.  Reuse exp(L dt) for every complete
        // exponential step; only a genuinely shorter final interval needs a
        // second matrix exponential.
        arma::cx_mat fullStepPropagator;
        if(plan.propagation==SSPropagation::Exponential)
            fullStepPropagator=arma::expmat(arma::cx_mat(prepared.generator)*plan.timeStep);

        double t=0.0;
        for(size_t step=1;step<=steps;++step)
        {
            const double target=(step==steps)?plan.totalTime:
                std::min(plan.totalTime,static_cast<double>(step)*plan.timeStep);
            const double h=target-t;
            if(!(h>0.0)){error="SSGeneral produced a non-increasing propagation time grid";return false;}
            if(plan.propagation==SSPropagation::RK4)
                state=RK4(prepared.generator,state,h);
            else if(std::abs(h-plan.timeStep)<=timeTolerance)
                state=fullStepPropagator*state;
            else
                state=arma::expmat(arma::cx_mat(prepared.generator)*h)*state;
            if(!state.is_finite()){error="SSGeneral propagation produced a non-finite state";return false;}
            t=target;trajectory.times.push_back(t);trajectory.states.push_back(state);
        }
        return true;
    }
    bool SSPropagator::SolveTimeIntegrated(const SSExecutionPlan &plan,const SSPreparedCalculation &prepared,arma::cx_vec &integrated,std::string &error)
    {
        error.clear();integrated.reset();const arma::cx_vec rhs=-prepared.initialState;bool ok=false;
        try{ok=arma::spsolve(integrated,prepared.generator,rhs,"superlu");}catch(...){ok=false;}
        if(!ok)ok=arma::solve(integrated,arma::cx_mat(prepared.generator),rhs,arma::solve_opts::no_approx);
        if(!ok||!integrated.is_finite()){error="failed to solve L X=-rho0; the SS system may contain a non-decaying subspace";return false;}
        const double residual=arma::norm(prepared.generator*integrated-rhs,2)/std::max(1.0,arma::norm(rhs,2));
        if(!std::isfinite(residual)||residual>plan.solverResidualTolerance){error="SSGeneral time-integrated solve failed residual tolerance";return false;}
        return true;
    }
    bool SSPropagator::SolveSteadyState(const SSExecutionPlan &plan,const SSPreparedCalculation &prepared,arma::cx_vec &steady,std::string &error)
    {
        error.clear();steady.reset();
        const arma::cx_vec traceDrift=prepared.generator.t()*prepared.traceFunctional;
        if(arma::norm(traceDrift,2)>plan.solverResidualTolerance*std::max(1.0,arma::norm(prepared.generator,"fro")))
        {error="SSGeneral steadystate requires a closed trace-preserving system; terminal sinks need explicit product manifolds in MultiSSGeneral";return false;}
        arma::cx_mat A(prepared.generator);arma::cx_vec b(A.n_rows,arma::fill::zeros);
        A.row(A.n_rows-1)=prepared.traceFunctional.t();b(A.n_rows-1)=1.0;
        if(!arma::solve(steady,A,b,arma::solve_opts::no_approx)||!steady.is_finite()){error="failed to solve normalized SSGeneral steady state";return false;}
        const double r=arma::norm(prepared.generator*steady,2);const arma::cx_double tr=arma::cdot(prepared.traceFunctional,steady);
        if(r>plan.solverResidualTolerance*std::max(1.0,arma::norm(prepared.generator,"fro"))||std::abs(tr-arma::cx_double(1,0))>10*plan.solverResidualTolerance)
        {error="SSGeneral steady-state solution failed stationarity/normalization residual";return false;}
        return true;
    }
}
