//////////////////////////////////////////////////////////////////////////////
// General cross-hierarchy physics equivalence tests.
//
// These tests are representation contracts rather than legacy-task parity
// tests.  The same physical one-system model must produce the same oriented
// Hilbert Hamiltonian, Liouville commutator, Haberkorn loss, density trajectory
// and State observables in HSGeneral, SSGeneral, and the one-sector limit of
// MultiSSGeneral.
//////////////////////////////////////////////////////////////////////////////
#include "HSHamiltonianBuilder.h"
#include "HSObservableCollector.h"
#include "HSPropagator.h"
#include "HSReactionRelaxation.h"
#include "HSStatePreparation.h"
#include "SSLiouvillianBuilder.h"
#include "SSObservableCollector.h"
#include "SSPropagator.h"
#include "SSSystemPreparation.h"
#include "MultiSSNetworkBuilder.h"
#include "MultiSSObservableCollector.h"
#include "MultiSSPropagator.h"
#include "Interaction.h"
#include "ObjectParser.h"
#include "Spin.h"
#include "SpinSystem.h"
#include "State.h"
#include "Transition.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>

namespace
{
    struct GeneralPhysicsFixture
    {
        SpinAPI::system_ptr system;
        SpinAPI::state_ptr up;
        SpinAPI::state_ptr down;
    };

    GeneralPhysicsFixture BuildGeneralPhysicsFixture(bool reaction,bool molecularFrames)
    {
        auto e=std::make_shared<SpinAPI::Spin>("E",
            "type=electron;spin=1/2;tensor=matrix(\"2.0 0.12 0.04;0.12 2.3 -0.07;0.04 -0.07 2.55\");");
        auto b0=std::make_shared<SpinAPI::Interaction>("B0",
            "type=zeeman;spins=E;field=0.0011 -0.0007 0.0043;ignoretensors=false;"
            "commonprefactor=false;prefactor=1.0;");
        auto up=std::make_shared<SpinAPI::State>("Up","spin(E)=|1/2>;");
        auto down=std::make_shared<SpinAPI::State>("Down","spin(E)=|-1/2>;");
        auto system=std::make_shared<SpinAPI::SpinSystem>("System");
        system->Add(e);system->Add(b0);system->Add(up);system->Add(down);
        if(reaction)
        {
            auto sink=std::make_shared<SpinAPI::Transition>("sink",
                "type=sink;sourcestate=Up;rate=0.17;",system);
            system->Add(sink);
        }
        std::string properties="initialstate=Up;initialstatecoherences=keep;";
        if(molecularFrames)
            properties += "initialstateframe=molecular;transitionstateframe=molecular;observablestateframe=molecular;";
        system->SetProperties(std::make_shared<MSDParser::ObjectParser>("properties",properties));
        if(!system->ValidateInteractions().empty())return {};
        if(!up->ParseFromSystem(*system)||!down->ParseFromSystem(*system))return {};
        const std::vector<SpinAPI::system_ptr> systems{system};
        if(reaction&&!system->ValidateTransitions(systems).empty())return {};
        return {system,up,down};
    }

    void BuildOrientations(bool rotated,RunSection::General::HS::HSOrientation &hs,
        RunSection::General::SS::SSOrientation &ss,
        RunSection::General::MultiSS::MultiSSOrientation &ms)
    {
        const double a=rotated?0.37:0.0;
        const double b=rotated?0.83:0.0;
        const double g=rotated?-0.29:0.0;
        hs.alpha=a;hs.beta=b;hs.gamma=g;hs.weight=1.0;
        SpinAPI::CreateZYZRotationMatrix(a,b,g,hs.frameToLab);
        ss.alpha=a;ss.beta=b;ss.gamma=g;ss.weight=1.0;ss.frameToLab=hs.frameToLab;
        ms.alpha=a;ms.beta=b;ms.gamma=g;ms.weight=1.0;ms.frameToLab=hs.frameToLab;
    }

    RunSection::General::HS::HSExecutionPlan MakeHSPlan(bool rotated,bool secular)
    {
        RunSection::General::HS::HSExecutionPlan plan;
        plan.dynamics=RunSection::General::HS::Dynamics::Static;
        plan.calculation=RunSection::General::HS::Calculation::TimeEvolution;
        plan.sampling=RunSection::General::HS::Sampling::Direct;
        plan.orientation=rotated?RunSection::General::HS::OrientationMode::Explicit:
            RunSection::General::HS::OrientationMode::Identity;
        plan.approximation=secular?SpinAPI::HamiltonianApproximation::Secular:
            SpinAPI::HamiltonianApproximation::Full;
        plan.hasH0List=true;plan.h0List={"B0"};
        plan.propagation=RunSection::General::HS::PropagationMethod::RK4;
        plan.observableMode=RunSection::General::HS::ObservableMode::StatePopulations;
        plan.totalTime=0.24;plan.timeStep=0.03;
        return plan;
    }

    RunSection::General::SS::SSHamiltonianMode MakeSSMode(bool secular)
    {
        return secular?RunSection::General::SS::SSHamiltonianMode::RotatedSecular:
            RunSection::General::SS::SSHamiltonianMode::RotatedFull;
    }

    bool Same(const arma::cx_mat &a,const arma::cx_mat &b,double tol=2.0e-11)
    {
        if(a.n_rows!=b.n_rows||a.n_cols!=b.n_cols)return false;
        const double scale=std::max({1.0,arma::norm(a,"fro"),arma::norm(b,"fro")});
        return arma::norm(a-b,"fro")<=tol*scale;
    }

    bool Same(const arma::cx_vec &a,const arma::cx_vec &b,double tol=2.0e-11)
    {
        if(a.n_elem!=b.n_elem)return false;
        const double scale=std::max({1.0,arma::norm(a,2),arma::norm(b,2)});
        return arma::norm(a-b,2)<=tol*scale;
    }

    bool BuildHSObjects(const GeneralPhysicsFixture &fixture,bool rotated,bool secular,
        const RunSection::General::HS::HSOrientation &orientation,
        RunSection::General::HS::HSExecutionPlan &plan,SpinAPI::SpinSpace &space,
        arma::sp_cx_mat &H,arma::sp_cx_mat &K,arma::cx_mat &rho,
        RunSection::General::HS::HSReactionRelaxation &reaction,
        RunSection::General::HS::HSRelaxationContext &relaxationContext,
        RunSection::General::HS::HSObservableCollector &observables,
        std::vector<arma::sp_cx_mat> &observableOperators,std::string &error)
    {
        plan=MakeHSPlan(rotated,secular);
        RunSection::General::HS::HSHamiltonianBuilder hbuilder(plan,space);
        if(!hbuilder.BuildStatic(orientation,H,nullptr,error))return false;

        std::mt19937 generator(12345);std::ostringstream log;
        RunSection::General::HS::HSPreparedState reference;
        if(!RunSection::General::HS::HSStatePreparation::Prepare(
            plan,fixture.system,space,reference,generator,log,error))return false;
        RunSection::General::HS::HSOrientedState oriented;
        if(!RunSection::General::HS::HSStatePreparation::PrepareForOrientation(
            plan,space,reference,orientation,oriented,error))return false;
        rho=oriented.density;

        if(!reaction.Validate(error)||!reaction.StaticReaction(orientation,K,error)||
            !reaction.PrepareRelaxation(orientation,H,relaxationContext,error))return false;
        if(!observables.Prepare(plan,fixture.system,space,log,error)||
            !observables.OperatorsForOrientation(space,orientation,observableOperators,error))return false;
        return true;
    }

    bool CheckHamiltonianPair(bool rotated,bool secular)
    {
        auto fixture=BuildGeneralPhysicsFixture(false,false);if(!fixture.system)return false;
        RunSection::General::HS::HSOrientation ho;RunSection::General::SS::SSOrientation so;
        RunSection::General::MultiSS::MultiSSOrientation mo;BuildOrientations(rotated,ho,so,mo);

        auto hp=MakeHSPlan(rotated,secular);SpinAPI::SpinSpace hsSpace(fixture.system);
        RunSection::General::HS::HSHamiltonianBuilder hb(hp,hsSpace);arma::sp_cx_mat Hhs;std::string error;
        if(!hb.BuildStatic(ho,Hhs,nullptr,error))return false;

        SpinAPI::SpinSpace ssSpace(fixture.system);arma::sp_cx_mat Hss;
        if(!RunSection::General::SS::SSLiouvillianBuilder::BuildHamiltonian(
            fixture.system,ssSpace,MakeSSMode(secular),so.frameToLab,Hss,error))return false;
        return Same(arma::cx_mat(Hhs),arma::cx_mat(Hss),1.0e-12);
    }

    bool CheckLiouvillianPair(bool rotated,bool secular)
    {
        auto fixture=BuildGeneralPhysicsFixture(false,false);if(!fixture.system)return false;
        RunSection::General::HS::HSOrientation ho;RunSection::General::SS::SSOrientation so;
        RunSection::General::MultiSS::MultiSSOrientation mo;BuildOrientations(rotated,ho,so,mo);
        SpinAPI::SpinSpace space(fixture.system);arma::sp_cx_mat H,L;std::string error;
        if(!RunSection::General::SS::SSLiouvillianBuilder::BuildHamiltonian(
            fixture.system,space,MakeSSMode(secular),so.frameToLab,H,error)||
           !RunSection::General::SS::SSLiouvillianBuilder::BuildInternalLiouvillian(
            fixture.system,space,H,so.frameToLab,L,error))return false;
        const arma::uword d=H.n_rows;
        const arma::sp_cx_mat I=arma::speye<arma::sp_cx_mat>(d,d);
        const arma::sp_cx_mat expected=arma::cx_double(0.0,-1.0)*
            (arma::kron(H,I)-arma::kron(I,H.st()));
        return Same(arma::cx_mat(L),arma::cx_mat(expected),1.0e-13);
    }

    bool CheckTrajectoryCase(bool reactionEnabled,bool rotated,bool secular,bool molecularFrames)
    {
        auto fixture=BuildGeneralPhysicsFixture(reactionEnabled,molecularFrames);if(!fixture.system)return false;
        RunSection::General::HS::HSOrientation ho;RunSection::General::SS::SSOrientation so;
        RunSection::General::MultiSS::MultiSSOrientation mo;BuildOrientations(rotated,ho,so,mo);
        std::string error;

        auto hp=MakeHSPlan(rotated,secular);SpinAPI::SpinSpace hsSpace(fixture.system);
        RunSection::General::HS::HSReactionRelaxation hsReaction(hp,fixture.system,hsSpace);
        RunSection::General::HS::HSRelaxationContext hsRelaxContext;
        RunSection::General::HS::HSObservableCollector hsObservables;
        std::vector<arma::sp_cx_mat> hsObservableOperators;
        arma::sp_cx_mat Hhs,Khs;arma::cx_mat rhoHS;
        if(!BuildHSObjects(fixture,rotated,secular,ho,hp,hsSpace,Hhs,Khs,rhoHS,
            hsReaction,hsRelaxContext,hsObservables,hsObservableOperators,error))return false;

        RunSection::General::SS::SSExecutionPlan sp;
        sp.calculation=RunSection::General::SS::SSCalculation::TimeEvolution;
        sp.propagation=RunSection::General::SS::SSPropagation::RK4;
        sp.observables=RunSection::General::SS::SSObservableMode::States;
        sp.orientation=rotated?RunSection::General::SS::SSOrientationMode::Explicit:
            RunSection::General::SS::SSOrientationMode::Identity;
        sp.hamiltonianMode=MakeSSMode(secular);sp.totalTime=hp.totalTime;sp.timeStep=hp.timeStep;
        RunSection::General::SS::SSPreparedCalculation ssPrepared;
        if(!RunSection::General::SS::SSSystemPreparation::Prepare(fixture.system,sp,so,ssPrepared,error))return false;
        RunSection::General::SS::SSTrajectory ssTrajectory;
        if(!RunSection::General::SS::SSPropagator::Propagate(sp,ssPrepared,ssTrajectory,error))return false;
        RunSection::General::SS::SSObservableCollector ssObservables;
        if(!ssObservables.Prepare(sp,ssPrepared,so,error))return false;

        RunSection::General::MultiSS::MultiSSExecutionPlan mp;
        mp.calculation=RunSection::General::MultiSS::MultiSSCalculation::TimeEvolution;
        mp.propagation=RunSection::General::MultiSS::MultiSSPropagation::RK4;
        mp.observables=RunSection::General::MultiSS::MultiSSObservableMode::States;
        mp.orientation=rotated?RunSection::General::MultiSS::MultiSSOrientationMode::Explicit:
            RunSection::General::MultiSS::MultiSSOrientationMode::Identity;
        mp.hamiltonianMode=MakeSSMode(secular);mp.totalTime=hp.totalTime;mp.timeStep=hp.timeStep;
        RunSection::General::MultiSS::MultiSSNetwork network;
        if(!RunSection::General::MultiSS::MultiSSNetworkBuilder::Build({fixture.system},mp,mo,network,error))return false;
        RunSection::General::MultiSS::MultiSSTrajectory multiTrajectory;
        if(!RunSection::General::MultiSS::MultiSSPropagator::Propagate(mp,network,multiTrajectory,error))return false;
        RunSection::General::MultiSS::MultiSSObservableCollector multiObservables;
        if(!multiObservables.Prepare(mp,network,mo,error))return false;

        if(!Same(arma::cx_mat(Hhs),arma::cx_mat(ssPrepared.local.hamiltonian),1.0e-12)||
           !Same(ssPrepared.initialState,network.systems.initialState,1.0e-12)||
           !Same(arma::cx_mat(ssPrepared.generator),arma::cx_mat(network.Generator(0.0)),1.0e-12))
            return false;
        if(ssTrajectory.states.size()!=multiTrajectory.states.size()||ssTrajectory.states.empty())return false;

        RunSection::General::HS::HSPropagator hsPropagator(hp,hsSpace);
        for(size_t step=0;step<ssTrajectory.states.size();++step)
        {
            if(step>0 && !hsPropagator.StepDensity(Hhs,Khs,hp.timeStep,rhoHS,
                hsReaction,hsRelaxContext,error))return false;
            arma::cx_mat rhoSS;
            ssPrepared.local.space->UseSuperoperatorSpace(true);
            if(!ssPrepared.local.space->OperatorFromSuperspace(ssTrajectory.states[step],rhoSS))return false;
            if(!Same(rhoHS,rhoSS,3.0e-10)||!Same(ssTrajectory.states[step],multiTrajectory.states[step],3.0e-10))return false;

            arma::rowvec hv,sv,mv;
            hsObservables.EvaluateDensity(hsObservableOperators,rhoHS,hv);
            if(!ssObservables.Evaluate(ssPrepared,ssTrajectory.states[step],sv,error)||
               !multiObservables.Evaluate(network,multiTrajectory.states[step],ssTrajectory.times[step],mv,error))return false;
            if(hv.n_elem!=sv.n_elem||hv.n_elem!=mv.n_elem)return false;
            if(arma::norm(hv-sv,2)>5.0e-10||arma::norm(hv-mv,2)>5.0e-10)return false;
        }

        // Trace/probability derivative must agree with the Hilbert Haberkorn
        // form at t=0. For a terminal sink, d Tr(rho)/dt=-2 Tr(K rho).
        arma::cx_vec rho0vec;
        ssPrepared.local.space->UseSuperoperatorSpace(true);
        if(!ssPrepared.local.space->OperatorToSuperspace(ssPrepared.local.initialDensity,rho0vec))return false;
        const arma::cx_double ssTraceDerivative=arma::cdot(ssPrepared.traceFunctional,ssPrepared.generator*rho0vec);
        const double hsTraceDerivative=-2.0*std::real(arma::trace(arma::cx_mat(Khs)*ssPrepared.local.initialDensity));
        if(std::abs(ssTraceDerivative.real()-hsTraceDerivative)>2.0e-12||std::abs(ssTraceDerivative.imag())>2.0e-12)return false;
        return true;
    }
}

bool test_general_physics_oriented_hamiltonian_equivalence()
{
    return CheckHamiltonianPair(false,false)&&CheckHamiltonianPair(true,false)&&
        CheckHamiltonianPair(false,true)&&CheckHamiltonianPair(true,true);
}

bool test_general_physics_row_major_commutator_equivalence()
{
    return CheckLiouvillianPair(false,false)&&CheckLiouvillianPair(true,false)&&
        CheckLiouvillianPair(false,true)&&CheckLiouvillianPair(true,true);
}

bool test_general_physics_hs_ss_multiss_trajectory_equivalence()
{
    // Full and secular, fixed and molecular State-frame semantics, with and
    // without terminal Haberkorn loss. Both identity and non-trivial explicit
    // orientation are exercised by real Hamiltonian propagation.
    for(bool reaction:{false,true})
        for(bool rotated:{false,true})
            for(bool secular:{false,true})
                for(bool molecular:{false,true})
                    if(!CheckTrajectoryCase(reaction,rotated,secular,molecular))return false;
    return true;
}

bool test_general_physics_state_frame_contract_is_backend_independent()
{
    // This focused gate specifically protects the fixed-vs-molecular semantic
    // boundary that used to be resolved independently in the three backends.
    return CheckTrajectoryCase(true,true,false,false)&&
        CheckTrajectoryCase(true,true,false,true);
}

void AddGeneralPhysicsEquivalenceTests(std::vector<test_case>&cases)
{
    cases.push_back({"General physics oriented HS/SS Hamiltonian equivalence",test_general_physics_oriented_hamiltonian_equivalence});
    cases.push_back({"General physics row-major Liouville commutator equivalence",test_general_physics_row_major_commutator_equivalence});
    cases.push_back({"General physics HS/SS/MultiSS trajectory equivalence",test_general_physics_hs_ss_multiss_trajectory_equivalence});
    cases.push_back({"General physics State-frame contract backend independence",test_general_physics_state_frame_contract_is_backend_independent});
}
