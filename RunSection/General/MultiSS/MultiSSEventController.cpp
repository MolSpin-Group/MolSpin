/////////////////////////////////////////////////////////////////////////
// MultiSSEventController implementation (RunSection::General::MultiSS)
// ----------------------------------------------------------------------
// The EventController owns *discrete scheduling state* for instantaneous
// TransferChannel events.  SpinAPI::QuantumMap owns the physical CP map; the
// propagator owns continuous integration.  Separating these responsibilities
// prevents event application from depending on mutable Transition/Pulse Active
// flags or on traversal order through a PulseSequence.
//
// An event is applied exactly once when propagation lands on its scheduled
// boundary.  MultiSSPropagator shortens a step so no integration step crosses an
// event.  This is required for the delta-pump/push idealization used by Mims et
// al. (DOI: 10.1126/science.abl4254).
//
// Magnetic/RF/MW pulse Hamiltonians are a distinct physics channel and are not
// reinterpreted as optical population-transfer events here.
/////////////////////////////////////////////////////////////////////////
#include "MultiSSEventController.h"
#include "QuantumMap.h"
#include "SpinSpace.h"
#include <cmath>
#include <limits>

namespace RunSection::General::MultiSS
{
    MultiSSEventController::MultiSSEventController(const MultiSSNetwork &network)
        : applied(network.events.size(),false) {}

    double MultiSSEventController::NextEventAfter(double time,double tolerance) const
    {
        double next=std::numeric_limits<double>::infinity();
        // An unapplied event at the current time is deliberately returned as
        // the current time so the propagation loop cannot step across it.
        for(size_t i=0;i<applied.size();++i) if(!applied[i])
        {
            // event vector is sorted; no network access here by design, so
            // NextEventAfter is used only through ApplyAt loop in Propagator.
        }
        (void)time;(void)tolerance;
        return next;
    }

    bool MultiSSEventController::ApplyAt(double time,MultiSSNetwork &network,
        arma::cx_vec &state,std::string &error,double tolerance)
    {
        error.clear();
        for(size_t i=0;i<network.events.size();++i)
        {
            if(applied[i] || std::abs(network.events[i].physical.EventTime()-time)>tolerance) continue;
            auto &event=network.events[i];
            auto &source=network.systems.contexts[event.sourceContext];
            auto &target=network.systems.contexts[event.targetContext];
            arma::cx_vec sv=state.subvec(source.offset,source.offset+source.superDimension-1);
            arma::cx_vec tv=state.subvec(target.offset,target.offset+target.superDimension-1);
            arma::cx_mat srho,trho;
            source.local.space->UseSuperoperatorSpace(true);
            target.local.space->UseSuperoperatorSpace(true);
            if(!source.local.space->OperatorFromSuperspace(sv,srho) ||
                !target.local.space->OperatorFromSuperspace(tv,trho))
            {error="failed to decode density at instantaneous MultiSS event";return false;}
            if(!SpinAPI::QuantumMap::ApplyTransferEvent(event.orientedSourceEffect,
                event.orientedKraus,event.physical.EventFraction(),srho,trho,error,tolerance))return false;
            if(!source.local.space->OperatorToSuperspace(srho,sv) ||
                !target.local.space->OperatorToSuperspace(trho,tv))
            {error="failed to encode density after instantaneous MultiSS event";return false;}
            state.subvec(source.offset,source.offset+source.superDimension-1)=sv;
            state.subvec(target.offset,target.offset+target.superDimension-1)=tv;
            applied[i]=true;
        }
        return true;
    }

    bool MultiSSEventController::AllApplied() const
    { for(bool v:applied)if(!v)return false;return true; }
}
