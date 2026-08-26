//////////////////////////////////////////////////////////////////////////////
// MolSpin Unit Testing Module
//
// Physics/numerics gates for the modular MultiSSGeneral architecture.
// These tests target the new reusable layers directly and do not call legacy
// tasks except where an explicit parity test is intended.
//////////////////////////////////////////////////////////////////////////////
#include "TimeProfile.h"
#include "TransferChannel.h"
#include "QuantumMap.h"
#include "NakajimaZwanzig.h"
#include "MultiSSExecutionPlan.h"
#include "MultiSSOrientationSampler.h"
#include "MultiSSNetworkBuilder.h"
#include "MultiSSPropagator.h"
#include "MultiSSObservableCollector.h"
#include "Spin.h"
#include "State.h"
#include "Transition.h"
#include "SpinSystem.h"
#include "ObjectParser.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <vector>

namespace
{
    using namespace RunSection::General::MultiSS;

    bool ValidateSystemObjects(const std::vector<SpinAPI::system_ptr> &systems)
    {
        bool ok=true;
        for(const auto &system:systems)
        {
            for(const auto &state:system->States()) ok &= state->ParseFromSystem(*system);
        }
        for(const auto &system:systems) ok &= system->ValidateTransitions(systems).empty();
        return ok;
    }

    SpinAPI::system_ptr ScalarSystem(const std::string &name,bool initial)
    {
        auto system=std::make_shared<SpinAPI::SpinSystem>(name);
        auto level=std::make_shared<SpinAPI::Spin>("level","spin=0;");
        auto state=std::make_shared<SpinAPI::State>("State","spin(level)=|0>;");
        system->Add(level);system->Add(state);
        system->SetProperties(std::make_shared<MSDParser::ObjectParser>(
            name+"_properties",initial?"initialstate=State;":""));
        return system;
    }

    double BlockTrace(const MultiSSNetwork &network,const arma::cx_vec &state,size_t index)
    {
        const auto &c=network.systems.contexts[index];
        arma::cx_mat rho;
        c.local.space->UseSuperoperatorSpace(true);
        c.local.space->OperatorFromSuperspace(state.subvec(c.offset,c.offset+c.superDimension-1),rho);
        return std::real(arma::trace(rho));
    }
}

bool test_multiss_timeprofile_gaussian_fraction_normalization()
{
    const double f=0.63;
    const double fwhm=7.25;
    const double peak=SpinAPI::GaussianTimeProfile::PeakForTransferredFraction(f,fwhm);
    SpinAPI::GaussianTimeProfile profile(12.0,fwhm,peak);
    const double area=profile.FullArea();
    const double recovered=1.0-std::exp(-area);
    const double numerical=profile.Integral(12.0-20.0*fwhm,12.0+20.0*fwhm);
    return std::abs(recovered-f)<2.0e-14 && std::abs(numerical-area)<2.0e-13;
}

bool test_multiss_preservespins_keeps_arbitrary_nuclear_coherence()
{
    auto s1=std::make_shared<SpinAPI::SpinSystem>("S1");
    auto css=std::make_shared<SpinAPI::SpinSystem>("CSS");
    auto shelf=std::make_shared<SpinAPI::Spin>("shelf","spin=0;");
    auto Ns=std::make_shared<SpinAPI::Spin>("N","spin=1;type=nucleus;");
    auto e1=std::make_shared<SpinAPI::Spin>("e1","spin=1/2;type=electron;");
    auto e2=std::make_shared<SpinAPI::Spin>("e2","spin=1/2;type=electron;");
    auto Nt=std::make_shared<SpinAPI::Spin>("N","spin=1;type=nucleus;");
    auto S1state=std::make_shared<SpinAPI::State>("S1state","spin(shelf)=|0>;");
    auto Singlet=std::make_shared<SpinAPI::State>("Singlet","spins(e1,e2)=|1/2,-1/2>-|-1/2,1/2>;");
    s1->Add(shelf);s1->Add(Ns);s1->Add(S1state);
    css->Add(e1);css->Add(e2);css->Add(Nt);css->Add(Singlet);
    auto transfer=std::make_shared<SpinAPI::Transition>("CS",
        "type=sink;sourcestate=S1state;target=CSS;targetstate=Singlet;rate=0.2;preservespins=N;",s1);
    s1->Add(transfer);
    s1->SetProperties(std::make_shared<MSDParser::ObjectParser>("p1","initialstate=S1state;"));
    css->SetProperties(std::make_shared<MSDParser::ObjectParser>("p2",""));
    std::vector<SpinAPI::system_ptr> systems{s1,css};
    if(!ValidateSystemObjects(systems)) return false;

    SpinAPI::TransferChannel channel;std::string error;
    if(!SpinAPI::TransferChannel::Compile(transfer,channel,error))
    {std::cout<<error<<std::endl;return false;}
    if(channel.KrausOperators().size()!=1 || channel.KrausOperators()[0].n_rows!=12 || channel.KrausOperators()[0].n_cols!=3)
        return false;

    arma::cx_mat rhoS(3,3,arma::fill::zeros);
    rhoS(0,0)=0.4;rhoS(1,1)=0.35;rhoS(2,2)=0.25;
    rhoS(0,1)=arma::cx_double(0.08,0.03);rhoS(1,0)=std::conj(rhoS(0,1));
    rhoS(1,2)=arma::cx_double(-0.04,0.02);rhoS(2,1)=std::conj(rhoS(1,2));
    arma::cx_mat rhoT(12,12,arma::fill::zeros);
    const arma::cx_mat before=rhoS;
    if(!SpinAPI::QuantumMap::ApplyTransferEvent(channel,1.0,rhoS,rhoT,error))
    {std::cout<<error<<std::endl;return false;}
    const arma::cx_mat C(channel.KrausOperators().front());
    const arma::cx_mat recovered=C.t()*rhoT*C;
    return arma::norm(rhoS,"fro")<1.0e-12 &&
        arma::norm(recovered-before,"fro")<1.0e-11 &&
        std::abs(std::real(arma::trace(rhoT))-1.0)<1.0e-12;
}

bool test_multiss_constant_transfer_matches_analytic_two_level_kinetics()
{
    auto A=ScalarSystem("A",true), B=ScalarSystem("B",false);
    auto tr=std::make_shared<SpinAPI::Transition>("AtoB",
        "type=sink;sourcestate=State;target=B;targetstate=State;rate=0.2;",A);
    A->Add(tr);std::vector<SpinAPI::system_ptr> systems{A,B};
    if(!ValidateSystemObjects(systems))return false;
    MultiSSExecutionPlan plan;plan.totalTime=2.0;plan.timeStep=0.01;plan.propagation=MultiSSPropagation::RK4;
    MultiSSOrientation o;SpinAPI::CreateZYZRotationMatrix(0,0,0,o.frameToLab);
    MultiSSNetwork network;std::string error;
    if(!MultiSSNetworkBuilder::Build(systems,plan,o,network,error)){std::cout<<error<<std::endl;return false;}
    MultiSSTrajectory trajectory;
    if(!MultiSSPropagator::Propagate(plan,network,trajectory,error)){std::cout<<error<<std::endl;return false;}
    const double pA=BlockTrace(network,trajectory.states.back(),0);
    const double pB=BlockTrace(network,trajectory.states.back(),1);
    const double ref=std::exp(-0.4);
    return std::abs(pA-ref)<2.0e-10 && std::abs(pB-(1.0-ref))<2.0e-10 && std::abs(pA+pB-1.0)<2.0e-11;
}

bool test_multiss_instantaneous_event_lands_exactly_and_applies_once()
{
    auto A=ScalarSystem("A",true),B=ScalarSystem("B",false);
    auto tr=std::make_shared<SpinAPI::Transition>("push",
        "type=sink;sourcestate=State;target=B;targetstate=State;rate=0;rateprofile=instantaneous;eventtime=0.5;transferfraction=0.4;",A);
    A->Add(tr);std::vector<SpinAPI::system_ptr> systems{A,B};
    if(!ValidateSystemObjects(systems))return false;
    MultiSSExecutionPlan plan;plan.totalTime=1.0;plan.timeStep=0.3;plan.propagation=MultiSSPropagation::RK4;
    MultiSSOrientation o;SpinAPI::CreateZYZRotationMatrix(0,0,0,o.frameToLab);
    MultiSSNetwork network;std::string error;if(!MultiSSNetworkBuilder::Build(systems,plan,o,network,error))return false;
    MultiSSTrajectory trajectory;if(!MultiSSPropagator::Propagate(plan,network,trajectory,error)){std::cout<<error<<std::endl;return false;}
    bool found=false;for(double t:trajectory.times)if(std::abs(t-0.5)<1.0e-14)found=true;
    const double pA=BlockTrace(network,trajectory.states.back(),0),pB=BlockTrace(network,trajectory.states.back(),1);
    return found && std::abs(pA-0.6)<1.0e-12 && std::abs(pB-0.4)<1.0e-12;
}

bool test_multiss_timeintegrated_is_not_steadystate()
{
    auto A=ScalarSystem("A",true),B=ScalarSystem("B",false);
    A->Add(std::make_shared<SpinAPI::Transition>("AtoB","type=sink;sourcestate=State;target=B;targetstate=State;rate=2;",A));
    B->Add(std::make_shared<SpinAPI::Transition>("sink","type=sink;sourcestate=State;rate=1;",B));
    std::vector<SpinAPI::system_ptr> systems{A,B};if(!ValidateSystemObjects(systems))return false;
    MultiSSExecutionPlan plan;plan.calculation=MultiSSCalculation::TimeIntegrated;
    MultiSSOrientation o;SpinAPI::CreateZYZRotationMatrix(0,0,0,o.frameToLab);
    MultiSSNetwork network;std::string error;if(!MultiSSNetworkBuilder::Build(systems,plan,o,network,error)){std::cout<<error<<std::endl;return false;}
    arma::cx_vec X;if(!MultiSSPropagator::SolveTimeIntegrated(plan,network,X,error)){std::cout<<error<<std::endl;return false;}
    return std::abs(BlockTrace(network,X,0)-0.5)<1.0e-11 && std::abs(BlockTrace(network,X,1)-1.0)<1.0e-11;
}

bool test_multiss_true_steadystate_closed_reversible_network()
{
    auto A=ScalarSystem("A",true),B=ScalarSystem("B",false);
    A->Add(std::make_shared<SpinAPI::Transition>("AtoB","type=sink;sourcestate=State;target=B;targetstate=State;rate=2;",A));
    B->Add(std::make_shared<SpinAPI::Transition>("BtoA","type=sink;sourcestate=State;target=A;targetstate=State;rate=1;",B));
    std::vector<SpinAPI::system_ptr> systems{A,B};if(!ValidateSystemObjects(systems))return false;
    MultiSSExecutionPlan plan;plan.calculation=MultiSSCalculation::SteadyState;
    MultiSSOrientation o;SpinAPI::CreateZYZRotationMatrix(0,0,0,o.frameToLab);
    MultiSSNetwork network;std::string error;if(!MultiSSNetworkBuilder::Build(systems,plan,o,network,error)){std::cout<<error<<std::endl;return false;}
    arma::cx_vec rho;if(!MultiSSPropagator::SolveSteadyState(plan,network,rho,error)){std::cout<<error<<std::endl;return false;}
    return std::abs(BlockTrace(network,rho,0)-1.0/3.0)<1.0e-11 && std::abs(BlockTrace(network,rho,1)-2.0/3.0)<1.0e-11;
}

bool test_multiss_historical_nz_core_exact_algebra()
{
    arma::vec eig={-0.3,0.7};arma::cx_mat omega,J,R;std::string error;
    if(!SpinAPI::NakajimaZwanzig::HistoricalFrequencyMatrix(eig,omega,&error))return false;
    if(!SpinAPI::NakajimaZwanzig::HistoricalExponentialSpectralDensity({0.8,0.0},{0.4,0.0},omega,J,&error))return false;
    arma::cx_mat op1=arma::cx_mat("0 1;1 0");
    arma::cx_mat op2=arma::cx_mat("1 0;0 -1");
    if(!SpinAPI::NakajimaZwanzig::HistoricalTensor(op1,op2,J,R,&error))return false;
    const arma::cx_mat I=arma::eye<arma::cx_mat>(2,2);
    const arma::cx_mat ref=-(arma::kron(op1.t(),I)-arma::kron(I,op1.t().st()))*J.t()*(arma::kron(op2,I)-arma::kron(I,op2.st()));
    return arma::norm(R-ref,"fro")<1.0e-14;
}


// Regression R3: three separately specified channels each consume population
// at their own rate.  Therefore three channels at k deplete the common source
// at 3k.  This prevents the factor-of-three ambiguity that can otherwise enter
// when three nuclear m_I target states are used as a proxy for one total rate.
bool test_multiss_three_parallel_channels_use_sum_of_rates()
{
    auto A=ScalarSystem("A",true);
    auto Bm=ScalarSystem("Bm",false), B0=ScalarSystem("B0",false), Bp=ScalarSystem("Bp",false);
    A->Add(std::make_shared<SpinAPI::Transition>("to_m",
        "type=sink;sourcestate=State;target=Bm;targetstate=State;rate=0.001;",A));
    A->Add(std::make_shared<SpinAPI::Transition>("to_0",
        "type=sink;sourcestate=State;target=B0;targetstate=State;rate=0.001;",A));
    A->Add(std::make_shared<SpinAPI::Transition>("to_p",
        "type=sink;sourcestate=State;target=Bp;targetstate=State;rate=0.001;",A));
    std::vector<SpinAPI::system_ptr> systems{A,Bm,B0,Bp};
    if(!ValidateSystemObjects(systems))return false;

    MultiSSExecutionPlan plan;plan.totalTime=200.0;plan.timeStep=0.2;plan.propagation=MultiSSPropagation::RK4;
    MultiSSOrientation o;SpinAPI::CreateZYZRotationMatrix(0,0,0,o.frameToLab);
    MultiSSNetwork network;std::string error;
    if(!MultiSSNetworkBuilder::Build(systems,plan,o,network,error)){std::cout<<error<<std::endl;return false;}
    MultiSSTrajectory trajectory;
    if(!MultiSSPropagator::Propagate(plan,network,trajectory,error)){std::cout<<error<<std::endl;return false;}
    const double ref=std::exp(-3.0*0.001*plan.totalTime);
    const double pA=BlockTrace(network,trajectory.states.back(),0);
    double products=0.0;for(size_t i=1;i<4;++i)products+=BlockTrace(network,trajectory.states.back(),i);
    return std::abs(pA-ref)<2.0e-11 && std::abs(products-(1.0-ref))<2.0e-11;
}

// Regression R6: the Mims et al. pump/push idealization is an instantaneous
// spin-selective quantum operation (Science 2021, DOI: 10.1126/science.abl4254).
// For separate singlet/triplet removal fractions fS and fT, an S/T coherence
// survives with sqrt[(1-fS)(1-fT)], not with a classical population factor.
bool test_multiss_mims_spin_selective_push_preserves_coherence_amplitude()
{
    arma::cx_mat rho(2,2,arma::fill::zeros), product(2,2,arma::fill::zeros);
    rho(0,0)=0.55;rho(1,1)=0.45;
    rho(0,1)=arma::cx_double(0.12,0.08);rho(1,0)=std::conj(rho(0,1));
    const arma::cx_mat before=rho;
    const double fS=0.64,fT=0.25;

    arma::sp_cx_mat GS(2,2),GT(2,2),CS(2,2),CT(2,2);
    GS(0,0)=1.0;GT(1,1)=1.0;CS(0,0)=1.0;CT(1,1)=1.0;
    std::string error;
    if(!SpinAPI::QuantumMap::ApplyTransferEvent(GS,{CS},fS,rho,product,error))
    {std::cout<<error<<std::endl;return false;}
    if(!SpinAPI::QuantumMap::ApplyTransferEvent(GT,{CT},fT,rho,product,error))
    {std::cout<<error<<std::endl;return false;}

    const arma::cx_double coherenceRef=
        std::sqrt((1.0-fS)*(1.0-fT))*before(0,1);
    return std::abs(rho(0,0)-(1.0-fS)*before(0,0))<1.0e-13 &&
        std::abs(rho(1,1)-(1.0-fT)*before(1,1))<1.0e-13 &&
        std::abs(rho(0,1)-coherenceRef)<1.0e-13 &&
        std::abs(product(0,0)-fS*before(0,0))<1.0e-13 &&
        std::abs(product(1,1)-fT*before(1,1))<1.0e-13 &&
        std::abs(std::real(arma::trace(rho)+arma::trace(product))-1.0)<1.0e-13;
}

// Regression R8: X=integral_0^inf rho(t) dt is a time-integrated transient,
// not a stationary state.  For one scalar sink at k the exact integral is 1/k;
// a sufficiently long explicit propagation must converge to the same number.
bool test_multiss_timeintegrated_matches_long_time_propagation()
{
    auto A=ScalarSystem("A",true);
    A->Add(std::make_shared<SpinAPI::Transition>("sink",
        "type=sink;sourcestate=State;rate=0.5;",A));
    std::vector<SpinAPI::system_ptr> systems{A};
    if(!ValidateSystemObjects(systems))return false;
    MultiSSExecutionPlan plan;plan.totalTime=30.0;plan.timeStep=0.01;plan.propagation=MultiSSPropagation::RK4;
    MultiSSOrientation o;SpinAPI::CreateZYZRotationMatrix(0,0,0,o.frameToLab);
    MultiSSNetwork network;std::string error;
    if(!MultiSSNetworkBuilder::Build(systems,plan,o,network,error))return false;
    arma::cx_vec X;if(!MultiSSPropagator::SolveTimeIntegrated(plan,network,X,error))return false;
    MultiSSTrajectory trajectory;if(!MultiSSPropagator::Propagate(plan,network,trajectory,error))return false;
    double numeric=0.0;
    for(size_t i=1;i<trajectory.times.size();++i)
    {
        const double dt=trajectory.times[i]-trajectory.times[i-1];
        numeric+=0.5*dt*(BlockTrace(network,trajectory.states[i-1],0)+
                         BlockTrace(network,trajectory.states[i],0));
    }
    const double exact=2.0;
    return std::abs(BlockTrace(network,X,0)-exact)<1.0e-11 &&
        std::abs(numeric-exact)<6.0e-6;
}

// Finite optical rates must conserve probability across represented source and
// destination manifolds at every time.  Gaussian k(t) is used here because the
// NV-centre rate-equation literature provides an explicit finite Gaussian
// optical-rate precedent (DOI: 10.1038/ncomms14000).
bool test_multiss_gaussian_transfer_conserves_trace_at_every_sample()
{
    auto A=ScalarSystem("A",true),B=ScalarSystem("B",false);
    A->Add(std::make_shared<SpinAPI::Transition>("pump",
        "type=sink;sourcestate=State;target=B;targetstate=State;rate=0;"
        "rateprofile=gaussian;pulsecenter=5;pulsefwhm=1;transferfraction=0.7;",A));
    std::vector<SpinAPI::system_ptr> systems{A,B};if(!ValidateSystemObjects(systems))return false;
    MultiSSExecutionPlan plan;plan.totalTime=10;plan.timeStep=0.01;plan.propagation=MultiSSPropagation::RK4;
    MultiSSOrientation o;SpinAPI::CreateZYZRotationMatrix(0,0,0,o.frameToLab);
    MultiSSNetwork network;std::string error;if(!MultiSSNetworkBuilder::Build(systems,plan,o,network,error))return false;
    MultiSSTrajectory trajectory;if(!MultiSSPropagator::Propagate(plan,network,trajectory,error))return false;
    for(const auto &x:trajectory.states)
        if(std::abs(BlockTrace(network,x,0)+BlockTrace(network,x,1)-1.0)>3.0e-11)return false;
    return std::abs(BlockTrace(network,trajectory.states.back(),1)-0.7)<2.0e-10;
}

// The immutable trajectory profile deliberately reproduces MolSpin's historical
// endpoint-hold + linear-interpolation convention without mutating Transition.
bool test_multiss_trajectory_timeprofile_interpolation_and_area()
{
    SpinAPI::TrajectoryTimeProfile p({0.0,1.0,3.0},{0.0,2.0,0.0});
    return std::abs(p.Value(0.5)-1.0)<1.0e-14 &&
        std::abs(p.Value(2.0)-1.0)<1.0e-14 &&
        std::abs(p.Value(-4.0))<1.0e-14 &&
        std::abs(p.Value(5.0))<1.0e-14 &&
        std::abs(p.Integral(0.0,3.0)-3.0)<1.0e-14;
}

// Short-pulse consistency gate: holding the integrated Gaussian action fixed
// must recover the same population redistribution as the instantaneous map.
// This is the numerical bridge between finite k(t) pulses and the delta-pulse
// event idealization used in DOI: 10.1126/science.abl4254.
bool test_multiss_short_gaussian_fixed_area_matches_instantaneous_event()
{
    auto Ag=ScalarSystem("Ag",true),Bg=ScalarSystem("Bg",false);
    Ag->Add(std::make_shared<SpinAPI::Transition>("pump",
        "type=sink;sourcestate=State;target=Bg;targetstate=State;rate=0;"
        "rateprofile=gaussian;pulsecenter=1;pulsefwhm=0.1;transferfraction=0.4;",Ag));
    std::vector<SpinAPI::system_ptr> sg{Ag,Bg};if(!ValidateSystemObjects(sg))return false;
    MultiSSExecutionPlan pg;pg.totalTime=2;pg.timeStep=0.001;pg.propagation=MultiSSPropagation::RK4;
    MultiSSOrientation o;SpinAPI::CreateZYZRotationMatrix(0,0,0,o.frameToLab);
    MultiSSNetwork ng;std::string error;if(!MultiSSNetworkBuilder::Build(sg,pg,o,ng,error))return false;
    MultiSSTrajectory tg;if(!MultiSSPropagator::Propagate(pg,ng,tg,error))return false;
    const double gaussianProduct=BlockTrace(ng,tg.states.back(),1);

    auto Ae=ScalarSystem("Ae",true),Be=ScalarSystem("Be",false);
    Ae->Add(std::make_shared<SpinAPI::Transition>("push",
        "type=sink;sourcestate=State;target=Be;targetstate=State;rate=0;"
        "rateprofile=instantaneous;eventtime=1;transferfraction=0.4;",Ae));
    std::vector<SpinAPI::system_ptr> se{Ae,Be};if(!ValidateSystemObjects(se))return false;
    MultiSSExecutionPlan pe;pe.totalTime=2;pe.timeStep=0.3;pe.propagation=MultiSSPropagation::RK4;
    MultiSSNetwork ne;if(!MultiSSNetworkBuilder::Build(se,pe,o,ne,error))return false;
    MultiSSTrajectory te;if(!MultiSSPropagator::Propagate(pe,ne,te,error))return false;
    const double eventProduct=BlockTrace(ne,te.states.back(),1);
    return std::abs(gaussianProduct-eventProduct)<2.0e-9 &&
        std::abs(eventProduct-0.4)<1.0e-12;
}

// Representation gate R5.  Steiner's reversible S1/CSS kinetic model
// (DOI: 10.1039/D6CP00916F) occupies the block-diagonal sector of an enlarged
// electronic Hilbert space when S1-CSS coupling is kinetic.  This test verifies
// algebraically that direct-sum Liouville evolution is exactly the restriction
// of the corresponding enlarged-Hilbert GKSL generator to that sector.
bool test_multiss_direct_sum_equals_enlarged_hilbert_block_diagonal_sector()
{
    const double k=0.37;
    arma::cx_mat Hcss(2,2,arma::fill::zeros);
    Hcss(0,0)=0.2;Hcss(1,1)=-0.15;Hcss(0,1)=arma::cx_double(0.07,-0.03);Hcss(1,0)=std::conj(Hcss(0,1));

    auto rowIndex=[](arma::uword r,arma::uword c,arma::uword d){return r*d+c;};
    arma::cx_mat Ld(5,5,arma::fill::zeros);
    for(arma::uword q=0;q<5;++q)
    {
        arma::cx_double a=0.0;arma::cx_mat rho(2,2,arma::fill::zeros);
        if(q==0)a=1.0;else{const arma::uword z=q-1;rho(z/2,z%2)=1.0;}
        const arma::cx_double da=-k*a;
        arma::cx_mat drho=arma::cx_double(0.0,-1.0)*(Hcss*rho-rho*Hcss);
        drho(0,0)+=k*a;
        Ld(0,q)=da;
        for(arma::uword r=0;r<2;++r)for(arma::uword c=0;c<2;++c)
            Ld(1+rowIndex(r,c,2),q)=drho(r,c);
    }

    arma::cx_mat H(3,3,arma::fill::zeros);H.submat(1,1,2,2)=Hcss;
    arma::cx_mat J(3,3,arma::fill::zeros);J(1,0)=std::sqrt(k);
    const arma::cx_mat JJ=J.t()*J;
    arma::cx_mat Lf(9,9,arma::fill::zeros);
    for(arma::uword q=0;q<9;++q)
    {
        arma::cx_mat rho(3,3,arma::fill::zeros);rho(q/3,q%3)=1.0;
        const arma::cx_mat drho=arma::cx_double(0.0,-1.0)*(H*rho-rho*H)+
            J*rho*J.t()-0.5*(JJ*rho+rho*JJ);
        for(arma::uword r=0;r<3;++r)for(arma::uword c=0;c<3;++c)
            Lf(rowIndex(r,c,3),q)=drho(r,c);
    }

    arma::cx_mat E(9,5,arma::fill::zeros);E(rowIndex(0,0,3),0)=1.0;
    for(arma::uword r=0;r<2;++r)for(arma::uword c=0;c<2;++c)
        E(rowIndex(r+1,c+1,3),1+rowIndex(r,c,2))=1.0;
    return arma::norm(Lf*E-E*Ld,"fro")<2.0e-14;
}


// Orientation/input-compatibility regression R10.  General/MultiSS reproduces
// the established system-level frame fallbacks (`transitionstateframe` and
// `observablestateframe`) without calling TaskMultiStaticSS.  A single common
// molecular->lab rotation must affect both the channel support and observable.
bool test_multiss_system_level_state_frames_use_common_orientation()
{
    auto A=std::make_shared<SpinAPI::SpinSystem>("A");
    auto B=std::make_shared<SpinAPI::SpinSystem>("B");
    auto ea=std::make_shared<SpinAPI::Spin>("e","spin=1/2;type=electron;");
    auto eb=std::make_shared<SpinAPI::Spin>("e","spin=1/2;type=electron;");
    auto upA=std::make_shared<SpinAPI::State>("Up","spin(e)=|1/2>;");
    auto upB=std::make_shared<SpinAPI::State>("Up","spin(e)=|1/2>;");
    A->Add(ea);A->Add(upA);B->Add(eb);B->Add(upB);
    A->SetProperties(std::make_shared<MSDParser::ObjectParser>("pa",
        "initialstate=Up;transitionstateframe=molecular;observablestateframe=molecular;"));
    B->SetProperties(std::make_shared<MSDParser::ObjectParser>("pb",
        "transitionstateframe=molecular;observablestateframe=molecular;"));
    A->Add(std::make_shared<SpinAPI::Transition>("AtoB",
        "type=sink;sourcestate=Up;target=B;targetstate=Up;rate=0.2;",A));
    std::vector<SpinAPI::system_ptr> systems{A,B};if(!ValidateSystemObjects(systems))return false;

    MultiSSExecutionPlan plan;plan.hamiltonianMode=RunSection::General::SS::SSHamiltonianMode::RotatedFull;
    plan.observables=MultiSSObservableMode::States;
    MultiSSOrientation o;o.alpha=0.37;o.beta=0.81;o.gamma=-0.22;o.weight=1.0;
    if(!SpinAPI::CreateZYZRotationMatrix(o.alpha,o.beta,o.gamma,o.frameToLab))return false;
    MultiSSNetwork network;std::string error;if(!MultiSSNetworkBuilder::Build(systems,plan,o,network,error))
    {std::cout<<error<<std::endl;return false;}
    if(network.continuousChannels.size()!=1)return false;

    SpinAPI::SpinSpace sa(A),sb(B);sa.UseSuperoperatorSpace(false);sb.UseSuperoperatorSpace(false);
    arma::cx_mat Ps,Pt,rotS,rotT;if(!sa.GetState(upA,Ps)||!sb.GetState(upB,Pt))return false;
    if(!sa.RotateState(Ps,o.frameToLab,rotS)||!sb.RotateState(Pt,o.frameToLab,rotT))return false;
    const auto &edge=network.continuousChannels.front();
    if(arma::norm(arma::cx_mat(edge.orientedSourceEffect)-rotS,"fro")>1.0e-11)return false;
    const arma::cx_mat C(edge.orientedKraus.front());
    if(arma::norm(C*C.t()-rotT,"fro")>1.0e-11)return false;

    // Put exactly the independently rotated B State into the direct-sum vector;
    // the collector must report B.Up.population=1 using the same orientation.
    arma::cx_vec global(network.systems.globalDimension,arma::fill::zeros),v;
    auto *bc=MultiSSSystemPreparation::FindContext(network.systems,B);if(bc==nullptr)return false;
    bc->local.space->UseSuperoperatorSpace(true);
    if(!bc->local.space->OperatorToSuperspace(rotT,v))return false;
    global.subvec(bc->offset,bc->offset+bc->superDimension-1)=v;
    MultiSSObservableCollector collector;if(!collector.Prepare(plan,network,o,error))return false;
    arma::rowvec values;if(!collector.Evaluate(network,global,0.0,values,error))return false;
    const auto &labels=collector.Labels();
    for(size_t i=0;i<labels.size();++i)
        if(labels[i]=="B.Up.population")return std::abs(values(i)-1.0)<1.0e-11;
    return false;
}

void AddMultiSSGeneralTests(std::vector<test_case> &cases)
{
    cases.push_back(test_case("MultiSS Gaussian rate fraction normalization",test_multiss_timeprofile_gaussian_fraction_normalization));
    cases.push_back(test_case("MultiSS preservespins preserves nuclear coherence",test_multiss_preservespins_keeps_arbitrary_nuclear_coherence));
    cases.push_back(test_case("MultiSS constant transfer analytic kinetics",test_multiss_constant_transfer_matches_analytic_two_level_kinetics));
    cases.push_back(test_case("MultiSS event boundary exact and once",test_multiss_instantaneous_event_lands_exactly_and_applies_once));
    cases.push_back(test_case("MultiSS timeintegrated solve semantics",test_multiss_timeintegrated_is_not_steadystate));
    cases.push_back(test_case("MultiSS true steady state",test_multiss_true_steadystate_closed_reversible_network));
    cases.push_back(test_case("MultiSS historical NZ core parity",test_multiss_historical_nz_core_exact_algebra));
    cases.push_back(test_case("MultiSS parallel-channel rates add",test_multiss_three_parallel_channels_use_sum_of_rates));
    cases.push_back(test_case("MultiSS Mims push coherence map",test_multiss_mims_spin_selective_push_preserves_coherence_amplitude));
    cases.push_back(test_case("MultiSS timeintegrated matches long propagation",test_multiss_timeintegrated_matches_long_time_propagation));
    cases.push_back(test_case("MultiSS Gaussian transfer conserves trace",test_multiss_gaussian_transfer_conserves_trace_at_every_sample));
    cases.push_back(test_case("MultiSS trajectory profile interpolation and area",test_multiss_trajectory_timeprofile_interpolation_and_area));
    cases.push_back(test_case("MultiSS short Gaussian matches event",test_multiss_short_gaussian_fixed_area_matches_instantaneous_event));
    cases.push_back(test_case("MultiSS direct sum equals enlarged block sector",test_multiss_direct_sum_equals_enlarged_hilbert_block_diagonal_sector));
    cases.push_back(test_case("MultiSS common orientation and frame fallbacks",test_multiss_system_level_state_frames_use_common_orientation));
}
