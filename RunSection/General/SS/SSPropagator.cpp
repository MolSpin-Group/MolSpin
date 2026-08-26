#include "SSPropagator.h"
#include <algorithm>
#include <cmath>
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
        arma::cx_vec state=prepared.initialState;trajectory.times.push_back(0.0);trajectory.states.push_back(state);double t=0.0;
        while(t<plan.totalTime-1.0e-14*std::max(1.0,plan.totalTime))
        {
            const double h=std::min(plan.timeStep,plan.totalTime-t);
            state=plan.propagation==SSPropagation::RK4?RK4(prepared.generator,state,h):arma::expmat(arma::cx_mat(prepared.generator)*h)*state;
            if(!state.is_finite()){error="SSGeneral propagation produced a non-finite state";return false;}
            t+=h;trajectory.times.push_back(t);trajectory.states.push_back(state);
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
