/////////////////////////////////////////////////////////////////////////
// MultiSSPropagator (RunSection::General::MultiSS)
// ------------------
// Numerical engine for the direct-sum master equation
//
//       d rho/dt = L(t) rho.
//
// It distinguishes three mathematically different calculations:
//   timeevolution  : explicit finite-time propagation;
//   timeintegrated : X=int_0^inf rho(t)dt, solving L X=-rho(0) for static L;
//   steadystate    : L rho_ss=0 with global trace one for a closed static network.
//
// `timeinf` is therefore NOT called a steady state internally.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_MultiSS_MultiSSPropagator
#define MOD_RunSection_General_MultiSS_MultiSSPropagator
#include <armadillo>
#include <string>
#include <vector>
#include "MultiSSExecutionPlan.h"
#include "MultiSSNetworkBuilder.h"
namespace RunSection::General::MultiSS
{
    struct MultiSSTrajectory
    {
        std::vector<double> times;
        std::vector<arma::cx_vec> states;
    };
    class MultiSSPropagator
    {
    public:
        static bool Propagate(const MultiSSExecutionPlan &_plan,MultiSSNetwork &_network,
            MultiSSTrajectory &_trajectory,std::string &_error);
        static bool SolveTimeIntegrated(const MultiSSExecutionPlan &_plan,const MultiSSNetwork &_network,
            arma::cx_vec &_integrated,std::string &_error);
        static bool SolveSteadyState(const MultiSSExecutionPlan &_plan,const MultiSSNetwork &_network,
            arma::cx_vec &_steady,std::string &_error);
    };
}
#endif
