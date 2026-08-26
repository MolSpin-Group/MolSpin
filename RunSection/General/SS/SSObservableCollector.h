/////////////////////////////////////////////////////////////////////////
// SSObservableCollector — State populations and terminal reaction observables.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_SS_SSObservableCollector
#define MOD_RunSection_General_SS_SSObservableCollector
#include "SSExecutionPlan.h"
#include "SSOrientationSampler.h"
#include "SSSystemPreparation.h"
#include <armadillo>
#include <string>
#include <vector>
namespace RunSection::General::SS
{
    struct SSObservable{std::string label;arma::cx_mat op;double scale=1.0;};
    class SSObservableCollector
    {
    public:
        bool Prepare(const SSExecutionPlan&,const SSPreparedCalculation&,const SSOrientation&,std::string&);
        const std::vector<std::string>&Labels()const{return labels;}
        bool Evaluate(const SSPreparedCalculation&,const arma::cx_vec&,arma::rowvec&,std::string&)const;
    private:
        std::vector<SSObservable> observables;std::vector<std::string> labels;
    };
}
#endif
