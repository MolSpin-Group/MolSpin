/////////////////////////////////////////////////////////////////////////
// MultiSSNetworkBuilder implementation (RunSection::General::MultiSS)
// ----------------------------------------------------------------------
// GRAPH COMPILATION
//   General/SS supplies diagonal one-manifold Liouvillians.  This file compiles
//   each parsed Transition exactly once into SpinAPI::TransferChannel and embeds
//   its source-loss and target-gain blocks in the direct-sum generator.
//
//       L_global(t) = blockdiag(L_i) + sum_c k_c(t) L_c^(unit).
//
//   That separation prevents the previous failure mode in which a local
//   reaction sink and an inter-system transfer edge both consume the same
//   population. Existing RunSection/Tasks remain independent references and
//   are never dispatched from this builder.
//
// PHYSICAL MAP
//   Each edge uses the CP kinetic structure documented in TransferChannel
//   (GKSL foundations: DOI: 10.1007/BF01608499, DOI: 10.1063/1.522979).
//   `preservespins` identity transport is particularly important for reversible
//   S1/CSS nuclear-memory models (DOI: 10.1039/D6CP00916F).
//
// ORIENTATION / INPUT COMPATIBILITY
//   Source and target State frames are resolved with the same precedence as the
//   established orientation-aware MultiStaticSS input contract: system-level
//   `transitionstateframe` is the fallback, while Transition-level sourceframe /
//   targetframe / transitionstateframe may override it.  The code reproduces
//   that input semantics locally; it does not call legacy task code.
//
// VECTORISATION
//   MolSpin uses its established row-major superspace convention.  Therefore
//   the target gain map C rho C^dagger is embedded as C (x) C*, while source
//   loss is 1/2[G rho + rho G].  The unit-rate edge is multiplied by k(t) only
//   in MultiSSNetwork::Generator().
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "MultiSSNetworkBuilder.h"
#include "../GeneralStateFrame.h"

#include "SpinSpace.h"
#include "SpinSystem.h"
#include "State.h"
#include "Transition.h"

#include <algorithm>
#include <cmath>

namespace RunSection::General::MultiSS
{
    namespace
    {
        bool RotationOperator(const MultiSSSystemContext &context,const arma::mat &rotation,
            arma::cx_mat &U,std::string &error)
        {
            const arma::uword d=context.hilbertDimension;
            SpinAPI::HilbertStateRotationCache cache;
            // CreateStateRotationCache always constructs Jx/Jy/Jz; identity is
            // used only as a dimension-correct seed. Its invariance flag does
            // not suppress CreateStateRotationOperator itself.
            if(!context.local.space->CreateStateRotationCache(
                arma::eye<arma::cx_mat>(d,d),cache) ||
                !context.local.space->CreateStateRotationOperator(rotation,cache,U))
            {error="failed to construct common spin-space rotation for \""+context.local.system->Name()+"\"";return false;}
            return true;
        }

        bool OrientPhysicalChannel(const MultiSSSystemContext &source,
            const MultiSSSystemContext *target,const MultiSSOrientation &orientation,
            SpinAPI::TransferChannel &physical,arma::sp_cx_mat &G,
            std::vector<arma::sp_cx_mat> &kraus,std::string &error)
        {
            G=physical.SourceEffect(); kraus=physical.KrausOperators();
            const auto t=physical.TransitionObject();
            const bool rotateS=::RunSection::General::TransitionSourceStateFrame(source.local.system,t)==SpinAPI::StateFrame::Molecular;
            const bool rotateT=target && ::RunSection::General::TransitionTargetStateFrame(target->local.system,t)==SpinAPI::StateFrame::Molecular;
            if(!rotateS&&!rotateT)return true;

            arma::cx_mat Us=arma::eye<arma::cx_mat>(source.hilbertDimension,source.hilbertDimension);
            arma::cx_mat Ut;
            if(rotateS&&!RotationOperator(source,orientation.frameToLab,Us,error))return false;
            if(target)
            {
                Ut=arma::eye<arma::cx_mat>(target->hilbertDimension,target->hilbertDimension);
                if(rotateT&&!RotationOperator(*target,orientation.frameToLab,Ut,error))return false;
            }

            if(physical.HasTarget())
            {
                std::vector<arma::sp_cx_mat> rotated;
                for(const auto &Csp:kraus)
                {
                    arma::cx_mat C(Csp);
                    if(rotateS) C=C*Us.t();
                    if(rotateT) C=Ut*C;
                    rotated.emplace_back(C);
                }
                kraus=std::move(rotated);
                arma::cx_mat denseG(source.hilbertDimension,source.hilbertDimension,arma::fill::zeros);
                for(const auto&Csp:kraus){arma::cx_mat C(Csp);denseG+=C.t()*C;}
                G=arma::sp_cx_mat(denseG);
            }
            else if(rotateS)
            {
                const arma::cx_mat dense=Us*arma::cx_mat(G)*Us.t();
                G=arma::sp_cx_mat(dense);
            }
            return true;
        }

        bool EmbedUnitGenerator(const MultiSSPreparedSystems &systems,
            const MultiSSSystemContext &source,const MultiSSSystemContext *target,
            const arma::sp_cx_mat &G,const std::vector<arma::sp_cx_mat> &kraus,
            arma::sp_cx_mat &global,std::string &error)
        {
            global.zeros(systems.globalDimension,systems.globalDimension);
            const arma::uword ds=source.hilbertDimension;
            const arma::sp_cx_mat I=arma::speye<arma::sp_cx_mat>(ds,ds);
            const arma::sp_cx_mat loss=0.5*(arma::kron(G,I)+arma::kron(I,G.st()));
            global.submat(source.offset,source.offset,
                source.offset+source.superDimension-1,source.offset+source.superDimension-1)-=loss;

            if(target)
            {
                arma::sp_cx_mat gain(target->superDimension,source.superDimension);
                for(const auto&C:kraus)
                    gain+=arma::kron(C,C.t().st());
                global.submat(target->offset,source.offset,
                    target->offset+target->superDimension-1,
                    source.offset+source.superDimension-1)+=gain;
            }
            return true;
        }
    }

    arma::sp_cx_mat MultiSSNetwork::Generator(double time) const
    {
        arma::sp_cx_mat L=systems.internalGenerator;
        for(const auto &channel:continuousChannels)
            L+=channel.physical.Rate(time)*channel.unitGlobalGenerator;
        return L;
    }

    bool MultiSSNetwork::IsTimeIndependent() const
    {
        for(const auto &c:continuousChannels) if(!c.physical.IsTimeIndependent())return false;
        return events.empty();
    }

    bool MultiSSNetwork::IsTracePreserving(double tolerance,double *residual) const
    {
        if(!IsTimeIndependent())return false;
        const arma::sp_cx_mat L=Generator(0.0);
        const arma::cx_rowvec left=systems.traceFunctional.t()*L;
        const double r=arma::norm(left,2);
        if(residual)*residual=r;
        return std::isfinite(r)&&r<=tolerance*std::max(1.0,arma::norm(arma::cx_mat(L),"fro"));
    }

    bool MultiSSNetworkBuilder::Build(const std::vector<SpinAPI::system_ptr> &systems,
        const MultiSSExecutionPlan &plan,const MultiSSOrientation &orientation,
        MultiSSNetwork &network,std::string &error)
    {
        network=MultiSSNetwork();error.clear();
        if(!MultiSSSystemPreparation::Prepare(systems,plan,orientation,network.systems,error))return false;

        for(size_t sourceIndex=0;sourceIndex<network.systems.contexts.size();++sourceIndex)
        {
            auto &source=network.systems.contexts[sourceIndex];
            for(const auto &transition:source.local.system->Transitions())
            {
                SpinAPI::TransferChannel physical;
                if(!SpinAPI::TransferChannel::Compile(transition,physical,error))
                {error="failed to compile Transition \""+transition->Name()+"\": "+error;return false;}

                MultiSSNetworkChannel compiled;
                compiled.physical=std::move(physical);
                compiled.sourceContext=sourceIndex;
                compiled.hasTarget=compiled.physical.HasTarget();
                MultiSSSystemContext *target=nullptr;
                if(compiled.hasTarget)
                {
                    target=MultiSSSystemPreparation::FindContext(network.systems,compiled.physical.TargetSystem());
                    if(!target){error="Transition \""+transition->Name()+"\" targets a SpinSystem outside MultiSSGeneral";return false;}
                    compiled.targetContext=static_cast<size_t>(target-&network.systems.contexts[0]);
                }

                if(!OrientPhysicalChannel(source,target,orientation,compiled.physical,
                    compiled.orientedSourceEffect,compiled.orientedKraus,error))return false;
                if(!compiled.physical.IsInstantaneous() &&
                    !EmbedUnitGenerator(network.systems,source,target,
                        compiled.orientedSourceEffect,compiled.orientedKraus,
                        compiled.unitGlobalGenerator,error))return false;

                if(compiled.physical.IsInstantaneous())
                {
                    if(!compiled.hasTarget){error="instantaneous sink events are not yet represented; add an explicit target manifold";return false;}
                    network.events.push_back(std::move(compiled));
                }
                else network.continuousChannels.push_back(std::move(compiled));
            }
        }
        std::sort(network.events.begin(),network.events.end(),[](const auto&a,const auto&b){return a.physical.EventTime()<b.physical.EventTime();});

        if(plan.IsStaticSolve()&&!network.IsTimeIndependent())
        {error="timeintegrated/steadystate requires constant continuous transfer rates and no instantaneous events";return false;}
        return true;
    }
}
