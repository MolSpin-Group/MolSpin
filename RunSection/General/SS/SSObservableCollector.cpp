/////////////////////////////////////////////////////////////////////////
// SSObservableCollector implementation (RunSection::General::SS)
// ------------------
// Constructs and evaluates state populations and terminal reaction yields.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// DEVELOPER WORKFLOW / OWNERSHIP MAP
// ----------------------------------------------------------------------
// SSGeneral observable construction and evaluation.
//
// What is done here:
//   - Prepares state-population and reaction/yield observables in one-manifold Liouville space.
//   - Evaluates observables from propagated or integrated superspace state vectors.
//
// Connections to the General framework / SpinAPI:
//   - Uses the prepared SpinAPI SpinSpace/state projectors from SSSystemPreparation.
//   - MultiSSObservableCollector extends the same idea to direct-sum manifold/network observables.
//
// Why this ownership is used:
//   - Observables are kept outside Liouvillian construction and propagation to preserve independent validation of dynamics and readout.
/////////////////////////////////////////////////////////////////////////

#include "SSObservableCollector.h"
#include "../GeneralStateFrame.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "State.h"
#include "Transition.h"
#include <cmath>

namespace RunSection::General::SS
{
    namespace
    {
        bool Real(const arma::cx_double &z,double &value,std::string &error,
            const std::string &label)
        {
            const double scale=std::max(1.0,std::abs(z.real()));
            if(!std::isfinite(z.real())||!std::isfinite(z.imag())||
                std::abs(z.imag())>1e-8*scale)
            {
                error="observable \""+label+"\" has a non-real value";
                return false;
            }
            value=std::abs(z.real())<1e-14?0.0:z.real();
            return true;
        }
    }

    bool SSObservableCollector::Prepare(const SSExecutionPlan &plan,
        const SSPreparedCalculation &prepared,const SSOrientation &orientation,
        std::string &error)
    {
        observables.clear();
        labels.clear();
        error.clear();
        const auto system=prepared.local.system;

        if(plan.observables==SSObservableMode::States||
            plan.observables==SSObservableMode::Both)
        {
            for(const auto &state:system->States())
            {
                if(!state)
                    continue;
                arma::cx_mat projector;
                if(!prepared.local.space->GetState(state,projector))
                {
                    error="failed to construct observable State \""+state->Name()+"\"";
                    return false;
                }
                const SpinAPI::StateFrame observableFrame=
                    ::RunSection::General::ObservableStateFrame(system,state);
                if(!::RunSection::General::ValidateProjectorStateFrame(observableFrame,
                    "observable State \""+state->Name()+"\"",error))return false;
                if(observableFrame==SpinAPI::StateFrame::Molecular&&
                    plan.orientation!=SSOrientationMode::Identity)
                {
                    arma::cx_mat rotated;
                    if(!prepared.local.space->RotateState(projector,orientation.frameToLab,rotated))
                    {
                        error="failed to rotate observable State \""+state->Name()+"\"";
                        return false;
                    }
                    projector=std::move(rotated);
                }
                SSObservable observable;
                // X=int rho(t)dt has units of time.  Its State projection is a
                // residence-time integral, not an instantaneous population.
                observable.label=system->Name()+"."+state->Name()+
                    (plan.calculation==SSCalculation::TimeIntegrated?
                        ".integrated_population_ns":".population");
                observable.op=std::move(projector);
                labels.push_back(observable.label);
                observables.push_back(std::move(observable));
            }
        }

        if(plan.observables==SSObservableMode::TransitionYields||
            plan.observables==SSObservableMode::Both)
        {
            for(const auto &transition:system->Transitions())
            {
                if(!transition||!transition->SourceState())
                    continue;
                arma::cx_mat projector;
                if(!prepared.local.space->GetState(transition->SourceState(),projector))
                {
                    error="failed to construct Transition observable \""+transition->Name()+"\"";
                    return false;
                }

                // Reaction/source-frame rotation was already validated in preparation.
                const SpinAPI::StateFrame sourceFrame=
                    ::RunSection::General::TransitionSourceStateFrame(system,transition);
                if(!::RunSection::General::ValidateProjectorStateFrame(sourceFrame,
                    "transition source State \""+transition->Name()+"\"",error))return false;
                const bool molecular=sourceFrame==SpinAPI::StateFrame::Molecular;

                if(molecular&&plan.orientation!=SSOrientationMode::Identity)
                {
                    arma::cx_mat rotated;
                    if(!prepared.local.space->RotateState(projector,orientation.frameToLab,rotated))
                    {
                        error="failed to rotate Transition observable \""+transition->Name()+"\"";
                        return false;
                    }
                    projector=std::move(rotated);
                }

                SSObservable observable;
                observable.label=system->Name()+"."+transition->Name()+
                    (plan.calculation==SSCalculation::TimeIntegrated?".yield":".flux");
                observable.op=std::move(projector);
                observable.scale=transition->Rate();
                labels.push_back(observable.label);
                observables.push_back(std::move(observable));
            }
        }
        return true;
    }

    bool SSObservableCollector::Evaluate(const SSPreparedCalculation &prepared,
        const arma::cx_vec &state,arma::rowvec &values,std::string &error)const
    {
        error.clear();
        arma::cx_mat rho;
        prepared.local.space->UseSuperoperatorSpace(true);
        if(!prepared.local.space->OperatorFromSuperspace(state,rho))
        {
            error="failed to decode SSGeneral density";
            return false;
        }
        values.zeros(observables.size());
        for(size_t i=0;i<observables.size();++i)
        {
            double value=0.0;
            if(!Real(arma::trace(observables[i].op*rho),value,error,observables[i].label))
                return false;
            values(i)=observables[i].scale*value;
        }
        return true;
    }
}
