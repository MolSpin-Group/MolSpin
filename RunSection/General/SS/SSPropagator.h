/////////////////////////////////////////////////////////////////////////
// SSPropagator — numerical engine for one static Liouville generator.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_SS_SSPropagator
#define MOD_RunSection_General_SS_SSPropagator
#include "SSExecutionPlan.h"
#include "SSSystemPreparation.h"
#include <armadillo>
#include <string>
#include <vector>
namespace RunSection::General::SS
{
    struct SSTrajectory{std::vector<double> times;std::vector<arma::cx_vec> states;};
    class SSPropagator
    {
    public:
        static bool Propagate(const SSExecutionPlan&,const SSPreparedCalculation&,SSTrajectory&,std::string&);
        static bool SolveTimeIntegrated(const SSExecutionPlan&,const SSPreparedCalculation&,arma::cx_vec&,std::string&);
        static bool SolveSteadyState(const SSExecutionPlan&,const SSPreparedCalculation&,arma::cx_vec&,std::string&);
    };
}
#endif
