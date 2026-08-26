/////////////////////////////////////////////////////////////////////////
// MultiSSObservableCollector (RunSection::General::MultiSS)
// ------------------
// Defines and evaluates observables on a direct-sum density vector.
//
// A manifold population is Tr(rho_i). A State population is Tr(P rho_i).
// Transfer flux is k(t) Tr(G rho_source).  Internal transfer fluxes are useful
// kinetic observables but are NOT terminal probabilities and therefore must
// not be summed with terminal yields as a normalization check.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_MultiSS_MultiSSObservableCollector
#define MOD_RunSection_General_MultiSS_MultiSSObservableCollector
#include <armadillo>
#include <string>
#include <vector>
#include "MultiSSExecutionPlan.h"
#include "MultiSSNetworkBuilder.h"
#include "MultiSSOrientationSampler.h"
namespace RunSection::General::MultiSS
{
    struct MultiSSObservable
    {
        std::string label;
        size_t context=0;
        arma::cx_mat operatorMatrix;
    };
    class MultiSSObservableCollector
    {
    public:
        bool Prepare(const MultiSSExecutionPlan &_plan,const MultiSSNetwork &_network,
            const MultiSSOrientation &_orientation,std::string &_error);
        const std::vector<std::string> &Labels() const { return labels; }
        bool Evaluate(const MultiSSNetwork &_network,const arma::cx_vec &_state,
            double _time,arma::rowvec &_values,std::string &_error) const;
    private:
        MultiSSExecutionPlan plan;
        std::vector<MultiSSObservable> observables;
        std::vector<std::string> labels;
    };
}
#endif
