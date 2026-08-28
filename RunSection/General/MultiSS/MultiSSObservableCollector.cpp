/////////////////////////////////////////////////////////////////////////
// MultiSSObservableCollector implementation (RunSection::General::MultiSS)
// ----------------------------------------------------------------------
// OBSERVABLE OWNERSHIP
//   This layer converts the global direct-sum density into experimental/model
//   observables.  It does not alter dynamics.  The primary quantities are
//     * manifold population        Tr rho_i,
//     * State/subspace population  Tr(P_a rho_i),
//     * instantaneous channel flux k(t) Tr(G_c rho_source).
//
//   These are the quantities needed to compare electronic-species population
//   decays and delayed-fluorescence kinetics, including the collaborator model
//   discussed in DOI: 10.1039/D6SC02081J.  A fluorescence intensity can, for
//   example, be formed from an S1 population as I_F(t)=k_F P_S1(t) when k_F is
//   the relevant radiative rate; the collector deliberately reports the more
//   primitive population/flux rather than baking in experiment-specific scaling.
//
// ORIENTATION CONTRACT
//   `observablestateframe` on a SpinSystem supplies the default State frame and
//   State-level `observableframe` may override it, matching the established
//   orientation-aware input semantics without depending on legacy task code.
//
// INTERNAL VS TERMINAL FLUX
//   Internal transfer fluxes are useful diagnostics but are not independent
//   product probabilities.  Summing every internal edge can exceed unity in a
//   network with recrossing.  Probability balance must instead use represented
//   manifold population plus true terminal product yields.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// DEVELOPER WORKFLOW / OWNERSHIP MAP
// ----------------------------------------------------------------------
// MultiSS network observable collection.
//
// What is done here:
//   - Builds labels and projectors for per-manifold populations/state populations and transition flux/yield observables.
//   - Evaluates observables from slices of the global direct-sum Liouville vector.
//
// Connections to the General framework / SpinAPI:
//   - Uses MultiSSSystemPreparation offsets and local SpinAPI SpinSpace objects.
//   - Mirrors SSObservableCollector semantics inside each manifold while adding network-level flux accounting.
//
// Why this ownership is used:
//   - Network observables are derived after propagation so kinetics and measurement definitions remain separate.
/////////////////////////////////////////////////////////////////////////

#include "MultiSSObservableCollector.h"
#include "../GeneralStateFrame.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "State.h"
#include "Transition.h"
#include <cmath>

namespace RunSection::General::MultiSS
{
    namespace
    {
        bool RealValue(const arma::cx_double &z,double &out,std::string &error,const std::string &label)
        {
            const double scale=std::max(1.0,std::abs(z.real()));
            if(!std::isfinite(z.real())||!std::isfinite(z.imag())||std::abs(z.imag())>1.0e-8*scale)
            {error="observable \""+label+"\" has a non-real value";return false;}
            out=std::abs(z.real())<1.0e-14?0.0:z.real();return true;
        }
    }

    bool MultiSSObservableCollector::Prepare(const MultiSSExecutionPlan &input,
        const MultiSSNetwork &network,const MultiSSOrientation &orientation,std::string &error)
    {
        plan=input;observables.clear();labels.clear();error.clear();
        for(size_t ci=0;ci<network.systems.contexts.size();++ci)
        {
            const auto &context=network.systems.contexts[ci];
            if(plan.observables==MultiSSObservableMode::Populations||plan.observables==MultiSSObservableMode::Both)
            {
                MultiSSObservable o;o.context=ci;o.label=context.local.system->Name()+
                    (plan.calculation==MultiSSCalculation::TimeIntegrated?
                        ".integrated_population_ns":".population");
                o.operatorMatrix=arma::eye<arma::cx_mat>(context.hilbertDimension,context.hilbertDimension);
                labels.push_back(o.label);observables.push_back(std::move(o));
            }
            if(plan.observables==MultiSSObservableMode::States||plan.observables==MultiSSObservableMode::Both)
            {
                for(const auto &state:context.local.system->States())
                {
                    if(state==nullptr)continue;
                    arma::cx_mat P;
                    if(!context.local.space->GetState(state,P))
                    {error="failed to construct observable State \""+state->Name()+"\"";return false;}
                    const SpinAPI::StateFrame observableFrame=
                        ::RunSection::General::ObservableStateFrame(context.local.system,state);
                    if(!::RunSection::General::ValidateProjectorStateFrame(observableFrame,
                        "observable State \""+state->Name()+"\"",error))return false;
                    if(observableFrame==SpinAPI::StateFrame::Molecular &&
                        input.orientation!=MultiSSOrientationMode::Identity)
                    {
                        arma::cx_mat rotated;
                        if(!context.local.space->RotateState(P,orientation.frameToLab,rotated))
                        {error="failed to rotate State observable \""+state->Name()+"\"";return false;}
                        P=std::move(rotated);
                    }
                    MultiSSObservable o;o.context=ci;o.label=context.local.system->Name()+"."+state->Name()+
                        (plan.calculation==MultiSSCalculation::TimeIntegrated?
                            ".integrated_population_ns":".population");o.operatorMatrix=std::move(P);
                    labels.push_back(o.label);observables.push_back(std::move(o));
                }
            }
        }
        if(plan.transitionFluxes)
        {
            for(const auto &c:network.continuousChannels)
            {
                // An internal channel may be traversed repeatedly, so its time
                // integral is an integrated event flux rather than an
                // independent terminal-product probability.
                labels.push_back(c.physical.SourceSystem()->Name()+"."+
                    c.physical.TransitionObject()->Name()+
                    (plan.calculation==MultiSSCalculation::TimeIntegrated?
                        ".integrated_flux":".flux"));
            }
        }
        return true;
    }

    bool MultiSSObservableCollector::Evaluate(const MultiSSNetwork &network,
        const arma::cx_vec &state,double time,arma::rowvec &values,std::string &error) const
    {
        error.clear();values.zeros(labels.size());size_t column=0;
        std::vector<arma::cx_mat> densities(network.systems.contexts.size());
        for(size_t ci=0;ci<network.systems.contexts.size();++ci)
        {
            const auto &context=network.systems.contexts[ci];
            const arma::cx_vec block=state.subvec(context.offset,context.offset+context.superDimension-1);
            context.local.space->UseSuperoperatorSpace(true);
            if(!context.local.space->OperatorFromSuperspace(block,densities[ci]))
            {error="failed to decode MultiSS density for observable evaluation";return false;}
        }
        for(const auto&o:observables)
        {
            double v=0.0;
            if(!RealValue(arma::trace(o.operatorMatrix*densities[o.context]),v,error,o.label))return false;
            values(column++)=v;
        }
        if(plan.transitionFluxes)
        {
            for(const auto &c:network.continuousChannels)
            {
                double population=0.0;
                const auto label=c.physical.SourceSystem()->Name()+"."+
                    c.physical.TransitionObject()->Name()+
                    (plan.calculation==MultiSSCalculation::TimeIntegrated?
                        ".integrated_flux":".flux");
                if(!RealValue(arma::trace(arma::cx_mat(c.orientedSourceEffect)*densities[c.sourceContext]),population,error,label))return false;
                values(column++)=c.physical.Rate(time)*population;
            }
        }
        return true;
    }
}
