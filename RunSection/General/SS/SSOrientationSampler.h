/////////////////////////////////////////////////////////////////////////
// SSOrientationSampler — common one-system molecular->lab orientation samples.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_SS_SSOrientationSampler
#define MOD_RunSection_General_SS_SSOrientationSampler
#include "SSExecutionPlan.h"
#include <armadillo>
#include <ostream>
#include <string>
#include <vector>
namespace RunSection::General::SS
{
    struct SSOrientation
    {
        double alpha=0.0,beta=0.0,gamma=0.0,weight=1.0;
        arma::mat frameToLab=arma::eye<arma::mat>(3,3);
    };
    class SSOrientationSampler
    {
    public:
        static bool Build(const SSExecutionPlan &_plan,std::vector<SSOrientation> &_out,
            std::ostream &_log,std::string &_error);
    };
}
#endif
