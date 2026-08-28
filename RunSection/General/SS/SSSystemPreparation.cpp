/////////////////////////////////////////////////////////////////////////
// SSSystemPreparation implementation.
// The local SSLiouvillianBuilder deliberately excludes reaction loss because
// MultiSS owns inter-manifold edges. SSGeneral adds only terminal one-system
// Haberkorn sinks here; inter-system transitions are rejected and delegated to
// MultiSSGeneral.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "SSSystemPreparation.h"
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
        bool IsIdentity(const arma::mat &R)
        {return R.n_rows==3&&R.n_cols==3&&arma::norm(R-arma::eye<arma::mat>(3,3),"fro")<1.0e-13;}
        arma::cx_vec TraceFunctional(SpinAPI::SpinSpace &space,arma::uword d)
        {
            arma::cx_vec out;space.UseSuperoperatorSpace(true);
            space.OperatorToSuperspace(arma::eye<arma::cx_mat>(d,d),out);return out;
        }
    }

    bool SSSystemPreparation::Prepare(const SpinAPI::system_ptr &system,const SSExecutionPlan &plan,
        const SSOrientation &orientation,SSPreparedCalculation &prepared,std::string &error)
    {
        prepared=SSPreparedCalculation();error.clear();
        if(!system){error="SSGeneral cannot prepare a null SpinSystem";return false;}
        if(system->InitialState().empty()){error="SSGeneral requires an initial state; empty destination manifolds belong to MultiSSGeneral";return false;}
        if(plan.reactionOperator!=SpinAPI::ReactionOperatorType::Haberkorn)
        {error="only Haberkorn reaction loss is currently qualified in SSGeneral";return false;}
        for(const auto&t:system->Transitions())
        {
            if(!t||!t->IsValid()||!t->SourceState()){error="SSGeneral encountered an invalid Transition/source State";return false;}
            if(t->Target()!=nullptr){error="inter-system Transition \""+t->Name()+"\" belongs to MultiSSGeneral";return false;}
            if(!SpinAPI::IsStatic(*t)){error="time-dependent Transition \""+t->Name()+"\" is not yet qualified in SSGeneral";return false;}
        }

        if(!SSLiouvillianBuilder::Prepare(system,plan.hamiltonianMode,orientation.frameToLab,
            prepared.local,error,plan.relaxationModel))return false;
        const arma::uword d=prepared.local.hamiltonian.n_rows;
        arma::cx_mat rho=prepared.local.initialDensity;
        const arma::cx_double tr=arma::trace(rho);
        if(!std::isfinite(tr.real())||!std::isfinite(tr.imag())||std::abs(tr.imag())>1.0e-10||!(tr.real()>0.0))
        {error="SSGeneral initial density has invalid/zero trace";return false;}
        rho/=tr.real(); prepared.local.initialDensity=rho;
        prepared.local.space->UseSuperoperatorSpace(true);
        if(!prepared.local.space->OperatorToSuperspace(rho,prepared.initialState))
        {error="failed to vectorize SSGeneral initial density";return false;}

        prepared.generator=prepared.local.internalLiouvillian;
        for(const auto&t:system->Transitions())
        {
            arma::cx_mat P;
            if(!prepared.local.space->GetState(t->SourceState(),P))
            {error="failed to construct reaction source State \""+t->SourceState()->Name()+"\"";return false;}
            const SpinAPI::StateFrame sourceFrame=
                ::RunSection::General::TransitionSourceStateFrame(system,t);
            if(!::RunSection::General::ValidateProjectorStateFrame(sourceFrame,
                "transition source State \""+t->Name()+"\"",error))return false;
            if(sourceFrame==SpinAPI::StateFrame::Molecular&&!IsIdentity(orientation.frameToLab))
            {
                arma::cx_mat rotated;
                if(!prepared.local.space->RotateState(P,orientation.frameToLab,rotated))
                {error="failed to rotate reaction source State \""+t->SourceState()->Name()+"\"";return false;}
                P=std::move(rotated);
            }
            const arma::sp_cx_mat Q=(t->Rate()/2.0)*arma::sp_cx_mat(P);
            arma::sp_cx_mat left,right;
            if(!prepared.local.space->SuperoperatorFromLeftOperator(Q,left)||
               !prepared.local.space->SuperoperatorFromRightOperator(Q,right))
            {error="failed to lift Haberkorn reaction operator \""+t->Name()+"\"";return false;}
            prepared.generator-=(left+right);
        }
        prepared.traceFunctional=TraceFunctional(*prepared.local.space,d);
        if(prepared.traceFunctional.n_elem!=d*d){error="failed to construct SSGeneral trace functional";return false;}
        return true;
    }
}
