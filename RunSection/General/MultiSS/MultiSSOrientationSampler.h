/////////////////////////////////////////////////////////////////////////
// MultiSSOrientationSampler (RunSection::General::MultiSS)
// ------------------
// Constructs one common molecular-to-laboratory orientation sample used by
// every SpinSystem in the network.  Independent per-manifold orientations are
// forbidden because they would destroy the covariance of a coupled molecular
// reaction network.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_MultiSS_MultiSSOrientationSampler
#define MOD_RunSection_General_MultiSS_MultiSSOrientationSampler
#include <armadillo>
#include <ostream>
#include <string>
#include <vector>
#include "MultiSSExecutionPlan.h"
namespace RunSection::General::MultiSS
{
    struct MultiSSOrientation
    {
        double alpha=0.0,beta=0.0,gamma=0.0,weight=1.0;
        arma::mat frameToLab;
    };
    class MultiSSOrientationSampler
    {
    public:
        static bool Build(const MultiSSExecutionPlan &_plan,
            std::vector<MultiSSOrientation> &_orientations,
            std::ostream &_log, std::string &_error);
    };
}
#endif
