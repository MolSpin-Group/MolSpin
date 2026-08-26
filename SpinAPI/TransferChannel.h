/////////////////////////////////////////////////////////////////////////
// TransferChannel (SpinAPI Module)
// ------------------
// Immutable compiled representation of kinetic population transfer between
// two Hilbert spaces.  This is the reusable physics primitive used by the new
// General/MultiSS layer; it is deliberately independent of RunSection tasks.
//
// HIERARCHY
//   parsed Transition  ->  SpinAPI::TransferChannel  -> General/MultiSS graph
//                                                       -> TaskMultiSSGeneral
//
// LEGACY BOUNDARY
//   Historical TaskMultiStaticSS and related task classes remain unchanged.
//   They continue to use their historical rank-one creation operators and are
//   regression references, not backends of this class.
//
// PHYSICS
//   A channel is represented by Kraus/jump maps C_mu : H_s -> H_t and
//
//       G = sum_mu C_mu^dagger C_mu,
//       d rho_s/dt = -k(t)/2 {G,rho_s},
//       d rho_t/dt = +k(t) sum_mu C_mu rho_s C_mu^dagger.
//
//   This is the block-diagonal (no inter-manifold coherence) restriction of a
//   completely-positive open-system transfer process.  The general GKSL
//   structure is grounded in
//       DOI: 10.1007/BF01608499
//       DOI: 10.1063/1.522979
//
// PRESERVED SPINS
//   The public Transition property is `preservespins`, *not* `spectators`.
//   A listed degree of freedom is NOT driven by the optical pulse.  It is
//   specifically carried through unchanged by an identity map.  For example,
//
//       C = |S_CSS><S1| (x) I_N
//
//   preserves every nuclear population and coherence during S1 <-> CSS
//   transfer.  This is the memory-preserving kinetic structure required by
//   reversible S1/radical-pair models such as
//       DOI: 10.1039/D6CP00916F
//
// Molecular Spin Dynamics Software.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_SpinAPI_TransferChannel
#define MOD_SpinAPI_TransferChannel

#include <armadillo>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "SpinAPIfwd.h"
#include "TimeProfile.h"

namespace SpinAPI
{
    class TransferChannel
    {
    public:
        TransferChannel() = default;

        static bool Compile(const transition_ptr &_transition,
            TransferChannel &_channel, std::string &_error,
            double _tolerance = 1.0e-10);

        const transition_ptr &TransitionObject() const { return transition; }
        const system_ptr &SourceSystem() const { return sourceSystem; }
        const system_ptr &TargetSystem() const { return targetSystem; }
        const state_ptr &SourceState() const { return sourceState; }
        const state_ptr &TargetState() const { return targetState; }
        const time_profile_ptr &Profile() const { return profile; }
        const std::vector<arma::sp_cx_mat> &KrausOperators() const { return kraus; }
        const arma::sp_cx_mat &SourceEffect() const { return sourceEffect; }
        const std::vector<std::pair<std::string, std::string>> &PreservedSpinMap() const { return preservedSpinMap; }

        bool HasTarget() const { return targetSystem != nullptr; }
        bool IsInstantaneous() const { return instantaneous; }
        double EventTime() const { return eventTime; }
        double EventFraction() const { return eventFraction; }
        double Rate(double _time) const { return profile ? profile->Value(_time) : 0.0; }
        bool IsTimeIndependent() const { return !instantaneous && profile && profile->IsConstant(); }

        // Unit-rate Liouville-space blocks.  Multiplication by k(t) is done by
        // the network layer so one immutable channel can be evaluated at any t.
        bool BuildSourceLossSuperoperator(arma::sp_cx_mat &_loss) const;
        bool BuildTargetGainSuperoperator(arma::sp_cx_mat &_gain) const;

    private:
        transition_ptr transition;
        system_ptr sourceSystem;
        system_ptr targetSystem;
        state_ptr sourceState;
        state_ptr targetState;
        time_profile_ptr profile;
        std::vector<arma::sp_cx_mat> kraus;
        arma::sp_cx_mat sourceEffect;
        std::vector<std::pair<std::string, std::string>> preservedSpinMap;
        bool instantaneous = false;
        double eventTime = 0.0;
        double eventFraction = 0.0;
    };
}

#endif
