/////////////////////////////////////////////////////////////////////////
// MultiSSEventController (RunSection::General::MultiSS)
// ------------------
// Pure timeline helper for discontinuous network events.
//
// It replaces the traversal-order-sensitive mutable Active flags used by some
// established PulseSequence workflows.  The propagator asks for the next event,
// lands on it exactly, and applies it exactly once.  Magnetic Pulse objects can
// be compiled into this same event timeline in a later extension without
// changing TransferChannel or QuantumMap.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_MultiSS_MultiSSEventController
#define MOD_RunSection_General_MultiSS_MultiSSEventController
#include <armadillo>
#include <string>
#include <vector>
#include "MultiSSNetworkBuilder.h"
namespace RunSection::General::MultiSS
{
    class MultiSSEventController
    {
    public:
        explicit MultiSSEventController(const MultiSSNetwork &_network);
        bool ApplyAt(double _time,MultiSSNetwork &_network,arma::cx_vec &_state,
            std::string &_error,double _tolerance=1.0e-10);
        bool AllApplied() const;
    private:
        std::vector<bool> applied;
    };
}
#endif
