/////////////////////////////////////////////////////////////////////////
// MultiSSNetworkBuilder (RunSection::General::MultiSS)
// ------------------
// Compiles parsed SpinAPI::Transition objects into immutable TransferChannels
// and embeds their source-loss / target-gain maps into the direct sum.
//
// IMPORTANT: a Transition appears exactly once in this layer. General/SS never
// adds Transition loss, so intersystem transfer cannot be double counted.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_MultiSS_MultiSSNetworkBuilder
#define MOD_RunSection_General_MultiSS_MultiSSNetworkBuilder

#include <armadillo>
#include <string>
#include <vector>
#include "MultiSSSystemPreparation.h"
#include "TransferChannel.h"

namespace RunSection::General::MultiSS
{
    struct MultiSSNetworkChannel
    {
        SpinAPI::TransferChannel physical;
        size_t sourceContext = 0;
        size_t targetContext = 0;
        bool hasTarget = false;
        arma::sp_cx_mat orientedSourceEffect;
        std::vector<arma::sp_cx_mat> orientedKraus;
        arma::sp_cx_mat unitGlobalGenerator;
    };

    struct MultiSSNetwork
    {
        MultiSSPreparedSystems systems;
        std::vector<MultiSSNetworkChannel> continuousChannels;
        std::vector<MultiSSNetworkChannel> events;

        arma::sp_cx_mat Generator(double _time) const;
        bool IsTimeIndependent() const;
        bool HasEvents() const { return !events.empty(); }
        bool IsTracePreserving(double _tolerance, double *_residual=nullptr) const;
    };

    class MultiSSNetworkBuilder
    {
    public:
        static bool Build(const std::vector<SpinAPI::system_ptr> &_systems,
            const MultiSSExecutionPlan &_plan,
            const MultiSSOrientation &_orientation,
            MultiSSNetwork &_network, std::string &_error);
    };
}
#endif
