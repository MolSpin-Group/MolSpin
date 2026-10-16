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
#include "FileReader.h"

#include <algorithm>
#include <cmath>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
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

    // Each test owns its input table.  In particular the cache-invalidation
    // check can rewrite one file without depending on the CTest working dir.
    class OpticalTestFile
    {
    public:
        explicit OpticalTestFile(const std::string &contents,
            const std::string &extension=".mst")
        {
            static std::atomic<unsigned long> serial{0};
            const auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
            path=std::filesystem::temp_directory_path()/
                ("molspin_optical_"+std::to_string(stamp)+"_"+
                 std::to_string(serial++)+extension);
            Write(contents);
        }
        ~OpticalTestFile()
        {
            std::error_code ignored;
            std::filesystem::remove(path,ignored);
        }
        OpticalTestFile(const OpticalTestFile &)=delete;
        OpticalTestFile &operator=(const OpticalTestFile &)=delete;
        bool Write(const std::string &contents) const
        {
            std::ofstream out(path,std::ios::trunc);
            out<<contents;
            return static_cast<bool>(out);
        }
        const std::filesystem::path &Path() const { return path; }
        std::string FileProperties() const
        {
            return "profilefile="+path.filename().string()+
                ";FileReaderDirectory="+path.parent_path().string()+"/;";
        }
    private:
        std::filesystem::path path;
    };

    bool CompileOptical(const OpticalTestFile &file,const std::string &options,
        SpinAPI::TransferChannel &channel,std::string &error)
    {
        auto A=ScalarSystem("A",true),B=ScalarSystem("B",false);
        auto tr=std::make_shared<SpinAPI::Transition>("pump",
            "type=sink;sourcestate=State;target=B;targetstate=State;"
            "rateprofile=experimental;"+file.FileProperties()+options,A);
        A->Add(tr);
        if(!ValidateSystemObjects({A,B}))
        {error="scalar test system failed Transition validation";return false;}
        return SpinAPI::TransferChannel::Compile(tr,channel,error);
    }

    bool PropagateOpticalScalar(const OpticalTestFile &file,
        const std::string &options,double step,double total,
        MultiSSPropagation method,double &source,double &target,
        std::vector<double> *times=nullptr)
    {
        auto A=ScalarSystem("A",true),B=ScalarSystem("B",false);
        A->Add(std::make_shared<SpinAPI::Transition>("pump",
            "type=sink;sourcestate=State;target=B;targetstate=State;"
            "rateprofile=experimental;"+file.FileProperties()+options,A));
        std::vector<SpinAPI::system_ptr> systems{A,B};
        if(!ValidateSystemObjects(systems))return false;
        MultiSSExecutionPlan plan;plan.totalTime=total;plan.timeStep=step;
        plan.propagation=method;plan.krylovTolerance=1.0e-12;
        MultiSSOrientation o;SpinAPI::CreateZYZRotationMatrix(0,0,0,o.frameToLab);
        MultiSSNetwork network;std::string error;
        if(!MultiSSNetworkBuilder::Build(systems,plan,o,network,error))
        {std::cout<<error<<std::endl;return false;}
        MultiSSTrajectory trajectory;
        if(!MultiSSPropagator::Propagate(plan,network,trajectory,error))
        {std::cout<<error<<std::endl;return false;}
        source=BlockTrace(network,trajectory.states.back(),0);
        target=BlockTrace(network,trajectory.states.back(),1);
        if(times)*times=trajectory.times;
        for(const auto &state:trajectory.states)
            if(std::abs(BlockTrace(network,state,0)+BlockTrace(network,state,1)-1.0)>2.0e-9)
                return false;
        return true;
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


bool test_multiss_krylov_handles_fast_transfer_without_rk4_stability_limit()
{
    auto A=ScalarSystem("A",true),B=ScalarSystem("B",false);
    A->Add(std::make_shared<SpinAPI::Transition>("fast",
        "type=sink;sourcestate=State;target=B;targetstate=State;rate=50;",A));
    std::vector<SpinAPI::system_ptr> systems{A,B};if(!ValidateSystemObjects(systems))return false;
    MultiSSExecutionPlan plan;plan.totalTime=1.0;plan.timeStep=1.0;
    plan.propagation=MultiSSPropagation::Krylov;plan.krylovDimension=30;plan.krylovTolerance=1.0e-12;
    MultiSSOrientation o;SpinAPI::CreateZYZRotationMatrix(0,0,0,o.frameToLab);
    MultiSSNetwork network;std::string error;if(!MultiSSNetworkBuilder::Build(systems,plan,o,network,error))return false;
    MultiSSTrajectory trajectory;if(!MultiSSPropagator::Propagate(plan,network,trajectory,error)){std::cout<<error<<std::endl;return false;}
    const double pA=BlockTrace(network,trajectory.states.back(),0),pB=BlockTrace(network,trajectory.states.back(),1);
    const double ref=std::exp(-50.0);
    return std::abs(pA-ref)<1.0e-11 && std::abs(pB-(1.0-ref))<1.0e-11;
}

bool test_multiss_exponential_allows_static_generator_with_exact_event_boundary()
{
    auto A=ScalarSystem("A",true),B=ScalarSystem("B",false);
    A->Add(std::make_shared<SpinAPI::Transition>("push",
        "type=sink;sourcestate=State;target=B;targetstate=State;rate=0;rateprofile=instantaneous;eventtime=0.5;transferfraction=0.4;",A));
    std::vector<SpinAPI::system_ptr> systems{A,B};if(!ValidateSystemObjects(systems))return false;
    MultiSSExecutionPlan plan;plan.totalTime=1.0;plan.timeStep=0.8;plan.propagation=MultiSSPropagation::Exponential;
    MultiSSOrientation o;SpinAPI::CreateZYZRotationMatrix(0,0,0,o.frameToLab);
    MultiSSNetwork network;std::string error;if(!MultiSSNetworkBuilder::Build(systems,plan,o,network,error))return false;
    MultiSSTrajectory trajectory;if(!MultiSSPropagator::Propagate(plan,network,trajectory,error)){std::cout<<error<<std::endl;return false;}
    bool found=false;for(double tt:trajectory.times)if(std::abs(tt-0.5)<1.0e-14)found=true;
    const double pA=BlockTrace(network,trajectory.states.back(),0),pB=BlockTrace(network,trajectory.states.back(),1);
    return found && std::abs(pA-0.6)<1.0e-12 && std::abs(pB-0.4)<1.0e-12;
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

bool test_multiss_nakajima_zwanzig_core_exact_algebra()
{
    arma::vec eig={-0.3,0.7};arma::cx_mat omega,J,R;std::string error;
    if(!SpinAPI::NakajimaZwanzig::FrequencyMatrix(eig,omega,&error))return false;
    if(!SpinAPI::NakajimaZwanzig::ExponentialSpectralDensity({0.8,0.0},{0.4,0.0},omega,J,&error))return false;
    arma::cx_mat op1=arma::cx_mat("0 1;1 0");
    arma::cx_mat op2=arma::cx_mat("1 0;0 -1");
    if(!SpinAPI::NakajimaZwanzig::RelaxationTensor(op1,op2,J,R,&error))return false;
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

    MultiSSExecutionPlan plan;
    plan.orientation=MultiSSOrientationMode::Explicit;
    plan.hamiltonianMode=RunSection::General::SS::SSHamiltonianMode::RotatedFull;
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

// The extra column is intentionally named `rate`: the strict optical reader
// selects only `laser`, while the historical trajectory API still sees `rate`.
bool test_multiss_experimental_opt_in_and_named_columns()
{
    OpticalTestFile file("time laser rate unrelated\n0 0 8 11\n1 2 8 12\n2 0 8 13\n");
    if(!std::filesystem::exists(file.Path()))return false;
    SpinAPI::TransferChannel channel;std::string error;
    if(!CompileOptical(file,
        "profilecolumn=laser;intensityunit=relative;ratelaw=linear;ratescale=0.2;",
        channel,error))
    {std::cout<<error<<std::endl;return false;}
    if(std::abs(channel.Rate(0.5)-0.2)>1.0e-13 ||
       std::abs(channel.Rate(1.0)-0.4)>1.0e-13 ||
       std::abs(channel.Profile()->Integral(0.0,2.0)-0.4)>1.0e-11)return false;

    SpinAPI::Trajectory legacy;
    if(!legacy.Load(file.Path().string()) || !legacy.HasColumn("rate") ||
       std::abs(legacy.Get(1,"rate")-8.0)>1.0e-13)return false;

    // A profilefile string alone cannot opt a Transition into optical I/O.
    // This spelling is rejected before attempting to open the missing file.
    auto A=ScalarSystem("A",true),B=ScalarSystem("B",false);
    auto tr=std::make_shared<SpinAPI::Transition>("no_opt_in",
        "type=sink;sourcestate=State;target=B;targetstate=State;"
        "rate=0.1;rateprofile=constant;profilefile=definitely_missing.mst;",A);
    A->Add(tr);
    if(!ValidateSystemObjects({A,B}))return false;
    SpinAPI::TransferChannel ordinary;
    if(SpinAPI::TransferChannel::Compile(tr,ordinary,error) ||
       error.find("profilefile")==std::string::npos)return false;
    SpinAPI::TransferChannel conflicting;
    return !CompileOptical(file,
        "profilecolumn=laser;intensityunit=relative;ratelaw=linear;ratescale=0.2;"
        "trajectory="+file.Path().filename().string()+";",conflicting,error);
}

bool test_multiss_experimental_absorption_codata_and_units()
{
    // Triangle area: 10^6 W ns cm^-2 = 10^-3 J cm^-2.
    OpticalTestFile file("time intensity\n0 0\n1 1000000\n2 0\n");
    constexpr double h=6.62607015e-34,c=299792458.0,NA=6.02214076e23;
    const double sigmaCm2=1000.0*std::log(10.0)*10000.0/NA;
    std::ostringstream sigmaText;sigmaText.precision(17);sigmaText<<sigmaCm2;
    const double expectedAction=sigmaCm2*1.0e-4*(1.0e-3/1.0e-4)*
        450.0e-9/(h*c);
    const std::string base="ratelaw=absorption;wavelength=450;";
    SpinAPI::TransferChannel epsilon,section,si,milli,kilo,mega;std::string error;
    if(!CompileOptical(file,base+
        "intensityunit=W/cm2;extinctioncoefficient=10000;",epsilon,error) ||
       !CompileOptical(file,base+
        "intensityunit=W/cm2;crosssection="+sigmaText.str()+";",
        section,error) ||
       !CompileOptical(file,base+
        "intensityunit=W/m2;intensityscale=10000;extinctioncoefficient=10000;",
        si,error) ||
       !CompileOptical(file,base+
        "intensityunit=\"mW/cm2\";intensityscale=1000;extinctioncoefficient=10000;",
        milli,error) ||
       !CompileOptical(file,base+
        "intensityunit=kW/cm2;intensityscale=0.001;extinctioncoefficient=10000;",
        kilo,error) ||
       !CompileOptical(file,base+
        "intensityunit=\"MW/cm2\";intensityscale=0.000001;extinctioncoefficient=10000;",
        mega,error))
    {std::cout<<error<<std::endl;return false;}
    const double area=epsilon.Profile()->Integral(0.0,2.0);
    const double peak=epsilon.Rate(1.0);
    return std::abs(sigmaCm2-3.8235325024087378e-17)<1.0e-29 &&
        std::abs(expectedAction-0.086616487426112387)<1.0e-14 &&
        std::abs(area-expectedAction)<2.0e-10 &&
        std::abs(peak-expectedAction)<2.0e-10 &&
        std::abs(section.Rate(1.0)-peak)<2.0e-10 &&
        std::abs(si.Rate(1.0)-peak)<2.0e-10 &&
        std::abs(milli.Rate(1.0)-peak)<2.0e-10 &&
        std::abs(kilo.Rate(1.0)-peak)<2.0e-10 &&
        std::abs(mega.Rate(1.0)-peak)<2.0e-10;
}

bool test_multiss_experimental_file_reader_preserves_irradiance_prefix_case()
{
    const std::vector<std::pair<std::string,std::string>> spellings={
        {"\"mW/cm2\"","mW/cm2"},
        {"\"MW/cm2\"","MW/cm2"},
        {"mw/cm2","mw/cm2"}
    };
    for(const auto &spelling:spellings)
    {
        OpticalTestFile input("SpinSystem A\n{\n"
            " Transition pump { intensityunit="+spelling.first+"; }\n"
            "}\n",".msd");
        MSDParser::FileReader reader(input.Path().string());
        if(!reader.IsOpen())return false;
        bool found=false;
        for(int item=0;item<30;++item)
        {
            const auto object=reader.ReadObject();
            if(object.Type()==MSDParser::ObjectType::EOFObject)break;
            if(object.Type()!=MSDParser::ObjectType::Transition)continue;
            MSDParser::ObjectParser properties(object.Name(),object.Contents());
            std::string parsed;
            if(!properties.Get("intensityunit",parsed) || parsed!=spelling.second)
            {std::cout<<"FileReader changed irradiance prefix case"<<std::endl;return false;}
            found=true;
        }
        if(!found)return false;
    }
    return true;
}

bool test_multiss_experimental_nonlinear_interpolate_intensity_first()
{
    OpticalTestFile file("time intensity\n0 0\n1 1\n2 0\n");
    SpinAPI::TransferChannel power,saturated;std::string error;
    if(!CompileOptical(file,
        "intensityunit=relative;ratelaw=powerlaw;ratescale=1;order=2;",
        power,error) ||
       !CompileOptical(file,
        "intensityunit=relative;ratelaw=saturating;ratescale=1;halfintensity=1;",
        saturated,error))
    {std::cout<<error<<std::endl;return false;}
    const double powerArea=2.0/3.0;
    const double saturationArea=2.0*(1.0-std::log(2.0));
    return std::abs(power.Rate(0.5)-0.25)<1.0e-13 &&
        std::abs(saturated.Rate(0.5)-1.0/3.0)<1.0e-13 &&
        std::abs(power.Profile()->Integral(0.0,2.0)-powerArea)<1.0e-9 &&
        std::abs(saturated.Profile()->Integral(0.0,2.0)-saturationArea)<1.0e-9 &&
        std::abs(power.Rate(-0.1))<1.0e-14 &&
        std::abs(saturated.Rate(2.1))<1.0e-14;
}

bool test_multiss_experimental_malformed_input_rejected()
{
    const std::string linear="intensityunit=relative;ratelaw=linear;ratescale=1;";
    const std::vector<std::pair<std::string,std::string>> badFiles={
        {"missing time header","clock intensity\n0 0\n1 1\n"},
        {"missing intensity header","time signal\n0 0\n1 1\n"},
        {"duplicate header","time intensity intensity\n0 0 0\n1 1 1\n"},
        {"malformed number","time intensity\n0 0\n1 wrong\n"},
        {"NaN","time intensity\n0 0\n1 NaN\n"},
        {"Inf","time intensity\n0 0\n1 Inf\n"},
        {"negative intensity","time intensity\n0 0\n1 -1\n"},
        {"decreasing time","time intensity\n0 0\n1 1\n0.5 0\n"},
        {"duplicate time","time intensity\n0 0\n1 1\n1 0\n"},
        {"truncated row","time intensity extra\n0 0 1\n1 1\n"},
        {"extra row field","time intensity\n0 0\n1 1 2\n"},
        {"one sample","time intensity\n0 1\n"}
    };
    for(const auto &item:badFiles)
    {
        OpticalTestFile file(item.second);
        SpinAPI::TransferChannel channel;std::string error;
        if(CompileOptical(file,linear,channel,error))
        {std::cout<<"accepted malformed optical file: "<<item.first<<std::endl;return false;}
    }
    OpticalTestFile good("time intensity\n0 0\n1 2\n2 0\n");
    const std::vector<std::pair<std::string,std::string>> badOptions={
        {"unknown time unit","profiletimeunit=fortnights;"+linear},
        {"unknown intensity unit","intensityunit=counts;ratelaw=linear;ratescale=1;"},
        {"ambiguous irradiance prefix","intensityunit=mw/cm2;ratelaw=absorption;wavelength=450;crosssection=1e-17;"},
        {"missing intensity unit","ratelaw=linear;ratescale=1;"},
        {"missing rate law","intensityunit=relative;ratescale=1;"},
        {"missing ratescale","intensityunit=relative;ratelaw=linear;"},
        {"missing order","intensityunit=relative;ratelaw=powerlaw;ratescale=1;"},
        {"invalid order","intensityunit=relative;ratelaw=powerlaw;ratescale=1;order=0;"},
        {"missing halfintensity","intensityunit=relative;ratelaw=saturating;ratescale=1;"},
        {"invalid halfintensity","intensityunit=relative;ratelaw=saturating;ratescale=1;halfintensity=0;"},
        {"relative absorption","intensityunit=relative;ratelaw=absorption;wavelength=450;crosssection=1e-17;"},
        {"missing wavelength","intensityunit=W/cm2;ratelaw=absorption;crosssection=1e-17;"},
        {"missing absorber","intensityunit=W/cm2;ratelaw=absorption;wavelength=450;"},
        {"two absorber parameters","intensityunit=W/cm2;ratelaw=absorption;wavelength=450;crosssection=1e-17;extinctioncoefficient=10000;"},
        {"ratescale on absorption","intensityunit=W/cm2;ratelaw=absorption;wavelength=450;crosssection=1e-17;ratescale=1;"},
        {"rate on experimental","intensityunit=relative;ratelaw=linear;ratescale=1;rate=0.2;"},
        {"lifetime on experimental","intensityunit=relative;ratelaw=linear;ratescale=1;lifetime=2;"},
        {"physical input on effective law","intensityunit=W/cm2;ratelaw=linear;ratescale=1;"},
        {"overflow","intensityunit=relative;ratelaw=linear;ratescale=1e308;intensityscale=1e308;"}
    };
    for(const auto &item:badOptions)
    {
        SpinAPI::TransferChannel channel;std::string error;
        if(CompileOptical(good,item.second,channel,error))
        {std::cout<<"accepted invalid optical parameters: "<<item.first<<std::endl;return false;}
    }
    OpticalTestFile missing("time intensity\n0 0\n1 1\n");
    std::error_code ignored;std::filesystem::remove(missing.Path(),ignored);
    SpinAPI::TransferChannel channel;std::string error;
    if(CompileOptical(missing,linear,channel,error))
    {std::cout<<"accepted missing optical file"<<std::endl;return false;}
    return true;
}

bool test_multiss_experimental_population_oracles_and_relative_amplitudes()
{
    OpticalTestFile file("time intensity\n0 0\n1 1\n2 0\n");
    for(double scale:{1.0,3.0})
    {
        const std::string options="intensityunit=relative;ratelaw=linear;"
            "ratescale=0.4;intensityscale="+std::to_string(scale)+";";
        SpinAPI::TransferChannel channel;std::string error;
        if(!CompileOptical(file,options,channel,error))
        {std::cout<<error<<std::endl;return false;}
        const double action=0.4*scale;
        if(std::abs(channel.Rate(0.5)-0.2*scale)>1.0e-12 ||
           std::abs(channel.Profile()->Integral(0.0,2.0)-action)>1.0e-10)
            return false;
        for(auto method:{MultiSSPropagation::RK4,MultiSSPropagation::Krylov})
        {
            double source=0.0,target=0.0;
            if(!PropagateOpticalScalar(file,options,0.025,3.0,method,source,target))
                return false;
            const double expected=std::exp(-action);
            if(std::abs(source-expected)>2.0e-8 ||
               std::abs(target-(1.0-expected))>2.0e-8)return false;
        }
    }
    // Population transfer is nonlinear in action: tripling rate does not
    // triple final product population once the source begins to deplete.
    const double p1=-std::expm1(-0.4),p3=-std::expm1(-1.2);
    return std::abs(p3-3.0*p1)>0.1;
}

bool test_multiss_experimental_nonlinear_population_oracles()
{
    OpticalTestFile file("time intensity\n0 0\n1 1\n2 0\n");
    const std::vector<std::pair<std::string,double>> laws={
        {"ratelaw=powerlaw;ratescale=0.4;order=2;",0.4*2.0/3.0},
        {"ratelaw=saturating;ratescale=0.4;halfintensity=1;",
            0.8*(1.0-std::log(2.0))}
    };
    for(const auto &law:laws)
    {
        const std::string options="intensityunit=relative;"+law.first;
        const double expected=std::exp(-law.second);
        for(auto method:{MultiSSPropagation::RK4,MultiSSPropagation::Krylov})
        {
            double source=0.0,target=0.0;
            if(!PropagateOpticalScalar(file,options,0.01,3.0,
                method,source,target))return false;
            const double tolerance=method==MultiSSPropagation::RK4?2.0e-7:1.0e-5;
            if(std::abs(source-expected)>tolerance ||
               std::abs(target-(1.0-expected))>tolerance)
            {
                std::cout<<"nonlinear population oracle: source="<<source
                    <<", expected="<<expected<<std::endl;
                return false;
            }
        }
    }
    return true;
}

bool test_multiss_experimental_shared_waveform_distinct_rates_and_cache_rewrite()
{
    OpticalTestFile file("time intensity\n0 0\n1 1\n2 0\n");
    auto A=ScalarSystem("A",true),B=ScalarSystem("B",false),C=ScalarSystem("C",false);
    const std::string common="type=sink;sourcestate=State;"
        "rateprofile=experimental;"+file.FileProperties()+
        "intensityunit=relative;ratelaw=linear;";
    A->Add(std::make_shared<SpinAPI::Transition>("toB",
        common+"target=B;targetstate=State;ratescale=0.2;",A));
    A->Add(std::make_shared<SpinAPI::Transition>("toC",
        common+"target=C;targetstate=State;ratescale=0.3;",A));
    std::vector<SpinAPI::system_ptr> systems{A,B,C};
    if(!ValidateSystemObjects(systems))return false;
    MultiSSExecutionPlan plan;plan.totalTime=3;plan.timeStep=0.025;
    plan.propagation=MultiSSPropagation::Krylov;plan.krylovTolerance=1.0e-12;
    MultiSSOrientation o;SpinAPI::CreateZYZRotationMatrix(0,0,0,o.frameToLab);
    MultiSSNetwork network;std::string error;
    if(!MultiSSNetworkBuilder::Build(systems,plan,o,network,error))
    {std::cout<<error<<std::endl;return false;}
    if(network.continuousChannels.size()!=2)return false;
    const auto &b=network.continuousChannels[0].physical;
    const auto &c=network.continuousChannels[1].physical;
    if(!b.OpticalProfile() || !c.OpticalProfile() ||
       b.OpticalProfile()->Waveform()!=c.OpticalProfile()->Waveform() ||
       std::abs(b.Rate(1.0)-0.2)>1.0e-13 ||
       std::abs(c.Rate(1.0)-0.3)>1.0e-13)return false;
    MultiSSTrajectory trajectory;
    if(!MultiSSPropagator::Propagate(plan,network,trajectory,error))
    {std::cout<<error<<std::endl;return false;}
    const auto &last=trajectory.states.back();
    const double e=std::exp(-0.5),moved=1.0-e;
    if(std::abs(BlockTrace(network,last,0)-e)>1.0e-9 ||
       std::abs(BlockTrace(network,last,1)-0.4*moved)>1.0e-9 ||
       std::abs(BlockTrace(network,last,2)-0.6*moved)>1.0e-9)
        return false;

    // A live cache must detect a changed file, while an independent new
    // compilation may read the new version. Existing channels remain immutable.
    SpinAPI::OpticalWaveformCache cache;
    std::shared_ptr<const SpinAPI::OpticalWaveform> firstWaveform,laterWaveform;
    if(!cache.Load(file.Path().string(),"intensity","ns",firstWaveform,error))return false;
    if(!file.Write("time intensity\n0 0\n1 10\n2 0\n"))return false;
    if(cache.Load(file.Path().string(),"intensity","ns",laterWaveform,error))
    {std::cout<<"cache silently reused changed optical file"<<std::endl;return false;}
    SpinAPI::TransferChannel revised;
    if(!CompileOptical(file,
        "intensityunit=relative;ratelaw=linear;ratescale=0.2;",revised,error))
    {std::cout<<error<<std::endl;return false;}
    return std::abs(b.Rate(1.0)-0.2)<1.0e-13 &&
        std::abs(revised.Rate(1.0)-2.0)<1.0e-13 &&
        revised.OpticalProfile()->Waveform()!=b.OpticalProfile()->Waveform();
}

bool test_multiss_experimental_nonzero_edges_event_order_and_time_units()
{
    OpticalTestFile ps("time intensity\n1000 1\n2000 1\n");
    SpinAPI::TransferChannel channel;std::string error;
    const std::string options="profiletimeunit=ps;intensityunit=relative;"
        "ratelaw=linear;ratescale=1;";
    if(!CompileOptical(ps,options,channel,error))
    {std::cout<<error<<std::endl;return false;}
    if(std::abs(channel.Rate(0.9))>1.0e-14 ||
       std::abs(channel.Rate(1.0)-1.0)>1.0e-14 ||
       std::abs(channel.Rate(2.0)-1.0)>1.0e-14 ||
       std::abs(channel.Rate(2.1))>1.0e-14 ||
       std::abs(channel.Profile()->Integral(0.0,3.0)-1.0)>1.0e-11)
        return false;
    double source=0.0,target=0.0;std::vector<double> times;
    if(!PropagateOpticalScalar(ps,options,0.7,3.0,
        MultiSSPropagation::Krylov,source,target,&times))return false;
    if(std::abs(source-std::exp(-1.0))>1.0e-10 ||
       std::abs(target+source-1.0)>1.0e-10)return false;
    double rkSource=0.0,rkTarget=0.0;
    if(!PropagateOpticalScalar(ps,options,0.1,3.0,
        MultiSSPropagation::RK4,rkSource,rkTarget))return false;
    if(std::abs(rkSource-std::exp(-1.0))>1.0e-6 ||
       std::abs(rkSource+rkTarget-1.0)>1.0e-9)return false;
    bool atStart=false,atEnd=false;
    for(double t:times){atStart|=std::abs(t-1.0)<1.0e-13;
                        atEnd|=std::abs(t-2.0)<1.0e-13;}
    if(!atStart||!atEnd)return false;

    // An event exactly at the leading optical edge acts once before the
    // continuous transfer on [1,2].
    auto A=ScalarSystem("A",true),B=ScalarSystem("B",false),C=ScalarSystem("C",false);
    A->Add(std::make_shared<SpinAPI::Transition>("pump",
        "type=sink;sourcestate=State;target=B;targetstate=State;"
        "rateprofile=experimental;"+ps.FileProperties()+options,A));
    A->Add(std::make_shared<SpinAPI::Transition>("push",
        "type=sink;sourcestate=State;target=C;targetstate=State;rate=0;"
        "rateprofile=instantaneous;eventtime=1;transferfraction=0.2;",A));
    std::vector<SpinAPI::system_ptr> systems{A,B,C};
    if(!ValidateSystemObjects(systems))return false;
    MultiSSExecutionPlan plan;plan.totalTime=3;plan.timeStep=0.7;
    plan.propagation=MultiSSPropagation::Krylov;
    MultiSSOrientation o;SpinAPI::CreateZYZRotationMatrix(0,0,0,o.frameToLab);
    MultiSSNetwork network;
    if(!MultiSSNetworkBuilder::Build(systems,plan,o,network,error))
    {std::cout<<error<<std::endl;return false;}
    MultiSSTrajectory trajectory;
    if(!MultiSSPropagator::Propagate(plan,network,trajectory,error))
    {std::cout<<error<<std::endl;return false;}
    const auto &last=trajectory.states.back();
    return std::abs(BlockTrace(network,last,0)-0.8*std::exp(-1.0))<1.0e-10 &&
        std::abs(BlockTrace(network,last,1)-0.8*(1.0-std::exp(-1.0)))<1.0e-10 &&
        std::abs(BlockTrace(network,last,2)-0.2)<1.0e-12;
}

bool test_multiss_experimental_diagnostics_and_legacy_layout()
{
    OpticalTestFile file("time intensity\n0 0\n1 1\n2 0\n");
    MultiSSOrientation o;SpinAPI::CreateZYZRotationMatrix(0,0,0,o.frameToLab);
    for(double scale:{1.0,3.0})for(bool optical:{false,true})for(bool allFluxes:{false,true})
    {
        auto A=ScalarSystem("A",true),B=ScalarSystem("B",false);
        A->Add(std::make_shared<SpinAPI::Transition>("pump",
            "type=sink;sourcestate=State;target=B;targetstate=State;"
            "rateprofile=experimental;"+file.FileProperties()+
            "intensityunit=relative;ratelaw=linear;ratescale=0.4;intensityscale="+
            std::to_string(scale)+";opticaldiagnostics="+
            std::string(optical?"true;":"false;"),A));
        std::vector<SpinAPI::system_ptr> systems{A,B};
        if(!ValidateSystemObjects(systems))return false;
        MultiSSExecutionPlan plan;plan.totalTime=0.5;plan.timeStep=0.5;
        plan.propagation=MultiSSPropagation::Krylov;
        plan.observables=MultiSSObservableMode::Populations;
        plan.transitionFluxes=allFluxes;
        MultiSSNetwork network;std::string error;
        if(!MultiSSNetworkBuilder::Build(systems,plan,o,network,error))
        {std::cout<<error<<std::endl;return false;}
        MultiSSObservableCollector collector;
        if(!collector.Prepare(plan,network,o,error))
        {std::cout<<error<<std::endl;return false;}
        const auto &labels=collector.Labels();
        const std::vector<std::string> oldPrefix={"A.population","B.population"};
        if(labels.size()<2 || labels[0]!=oldPrefix[0] || labels[1]!=oldPrefix[1])
            return false;
        auto count=[&labels](const std::string &name)
        {return static_cast<size_t>(std::count(labels.begin(),labels.end(),name));};
        if(!optical)
        {
            const std::vector<std::string> expected=allFluxes
                ?std::vector<std::string>{"A.population","B.population","A.pump.flux"}
                :oldPrefix;
            if(labels!=expected)return false;
        }
        else if(count("A.pump.intensity")!=1 ||
                count("A.pump.rate")!=1 ||
                count("A.pump.integratedrate")!=1 ||
                count("A.pump.flux")!=1)return false;

        MultiSSTrajectory trajectory;
        if(!MultiSSPropagator::Propagate(plan,network,trajectory,error))
        {std::cout<<error<<std::endl;return false;}
        arma::rowvec values;
        if(!collector.Evaluate(network,trajectory.states.back(),0.5,values,error))
        {std::cout<<error<<std::endl;return false;}
        const double population=std::exp(-0.05*scale);
        if(std::abs(values(0)-population)>1.0e-10 ||
           std::abs(values(1)-(1.0-population))>1.0e-10)return false;
        auto value=[&labels,&values](const std::string &name)
        {return values(static_cast<arma::uword>(std::find(labels.begin(),labels.end(),name)-labels.begin()));};
        if(optical && (std::abs(value("A.pump.intensity")-0.5)>1.0e-13 ||
           std::abs(value("A.pump.rate")-0.2*scale)>1.0e-13 ||
           std::abs(value("A.pump.integratedrate")-0.05*scale)>1.0e-10 ||
           std::abs(value("A.pump.flux")-0.2*scale*population)>1.0e-10))
            return false;
        if(!optical && allFluxes &&
            std::abs(value("A.pump.flux")-0.2*scale*population)>1.0e-10)
            return false;
    }
    return true;
}

bool test_multiss_experimental_preservespins_coherence()
{
    OpticalTestFile file("time intensity\n0 0\n1 1\n2 0\n");
    auto s1=std::make_shared<SpinAPI::SpinSystem>("S1");
    auto css=std::make_shared<SpinAPI::SpinSystem>("CSS");
    auto shelf=std::make_shared<SpinAPI::Spin>("shelf","spin=0;");
    auto Ns=std::make_shared<SpinAPI::Spin>("N","spin=1;type=nucleus;");
    auto e1=std::make_shared<SpinAPI::Spin>("e1","spin=1/2;type=electron;");
    auto e2=std::make_shared<SpinAPI::Spin>("e2","spin=1/2;type=electron;");
    auto Nt=std::make_shared<SpinAPI::Spin>("N","spin=1;type=nucleus;");
    auto S1state=std::make_shared<SpinAPI::State>("S1state","spin(shelf)=|0>;");
    auto Singlet=std::make_shared<SpinAPI::State>("Singlet",
        "spins(e1,e2)=|1/2,-1/2>-|-1/2,1/2>;");
    s1->Add(shelf);s1->Add(Ns);s1->Add(S1state);
    css->Add(e1);css->Add(e2);css->Add(Nt);css->Add(Singlet);
    auto transfer=std::make_shared<SpinAPI::Transition>("CS",
        "type=sink;sourcestate=S1state;target=CSS;targetstate=Singlet;"
        "preservespins=N;rateprofile=experimental;"+
        file.FileProperties()+"intensityunit=relative;ratelaw=linear;ratescale=0.4;",s1);
    s1->Add(transfer);
    s1->SetProperties(std::make_shared<MSDParser::ObjectParser>("p1","initialstate=S1state;"));
    css->SetProperties(std::make_shared<MSDParser::ObjectParser>("p2",""));
    std::vector<SpinAPI::system_ptr> systems{s1,css};
    if(!ValidateSystemObjects(systems))return false;
    MultiSSExecutionPlan plan;plan.totalTime=3;plan.timeStep=0.025;
    plan.propagation=MultiSSPropagation::Krylov;
    MultiSSOrientation o;SpinAPI::CreateZYZRotationMatrix(0,0,0,o.frameToLab);
    MultiSSNetwork network;std::string error;
    if(!MultiSSNetworkBuilder::Build(systems,plan,o,network,error))
    {std::cout<<error<<std::endl;return false;}
    if(network.continuousChannels.size()!=1)return false;
    const arma::cx_mat C(network.continuousChannels.front().physical.KrausOperators().front());
    arma::cx_mat rhoS(3,3,arma::fill::zeros);
    rhoS(0,0)=0.4;rhoS(1,1)=0.35;rhoS(2,2)=0.25;
    rhoS(0,1)=arma::cx_double(0.08,0.03);rhoS(1,0)=std::conj(rhoS(0,1));
    rhoS(1,2)=arma::cx_double(-0.04,0.02);rhoS(2,1)=std::conj(rhoS(1,2));
    arma::cx_vec sourceVector;
    auto &sourceContext=network.systems.contexts[0];
    sourceContext.local.space->UseSuperoperatorSpace(true);
    if(!sourceContext.local.space->OperatorToSuperspace(rhoS,sourceVector))return false;
    network.systems.initialState.zeros();
    network.systems.initialState.subvec(sourceContext.offset,
        sourceContext.offset+sourceContext.superDimension-1)=sourceVector;
    MultiSSTrajectory trajectory;
    if(!MultiSSPropagator::Propagate(plan,network,trajectory,error))
    {std::cout<<error<<std::endl;return false;}
    const auto &targetContext=network.systems.contexts[1];
    arma::cx_mat rhoT,remaining;
    if(!sourceContext.local.space->OperatorFromSuperspace(
        trajectory.states.back().subvec(sourceContext.offset,
            sourceContext.offset+sourceContext.superDimension-1),remaining))return false;
    targetContext.local.space->UseSuperoperatorSpace(true);
    if(!targetContext.local.space->OperatorFromSuperspace(
        trajectory.states.back().subvec(targetContext.offset,
            targetContext.offset+targetContext.superDimension-1),rhoT))return false;
    const double transferred=-std::expm1(-0.4);
    return arma::norm(remaining-std::exp(-0.4)*rhoS,"fro")<1.0e-9 &&
        arma::norm(C.t()*rhoT*C-transferred*rhoS,"fro")<1.0e-9 &&
        std::abs(std::real(arma::trace(remaining)+arma::trace(rhoT))-1.0)<1.0e-10;
}

bool test_multiss_experimental_sharp_pulse_timestep_convergence()
{
    OpticalTestFile file("time intensity\n0 0\n0.05 1\n0.1 0\n");
    const std::string options="intensityunit=relative;ratelaw=linear;ratescale=20;";
    const double exact=std::exp(-1.0);
    double coarse=0.0,middle=0.0,fine=0.0,target=0.0,krylov=0.0;
    if(!PropagateOpticalScalar(file,options,0.02,0.2,
        MultiSSPropagation::RK4,coarse,target) ||
       !PropagateOpticalScalar(file,options,0.01,0.2,
        MultiSSPropagation::RK4,middle,target) ||
       !PropagateOpticalScalar(file,options,0.005,0.2,
        MultiSSPropagation::RK4,fine,target) ||
       !PropagateOpticalScalar(file,options,0.005,0.2,
        MultiSSPropagation::Krylov,krylov,target))return false;
    const double ec=std::abs(coarse-exact),em=std::abs(middle-exact),ef=std::abs(fine-exact);
    return ef<em && em<ec && ef<2.0e-7 &&
        std::abs(krylov-exact)<1.0e-9 &&
        std::abs(krylov-fine)<2.0e-7;
}

bool test_multiss_experimental_nonlinear_krylov_timestep_convergence()
{
    OpticalTestFile file("time intensity\n0 0\n1 1\n2 0\n");
    const std::string options="intensityunit=relative;ratelaw=powerlaw;"
        "ratescale=0.4;order=2;";
    const double exact=std::exp(-0.4*2.0/3.0);
    double coarse=0.0,middle=0.0,fine=0.0,target=0.0;
    if(!PropagateOpticalScalar(file,options,0.1,2.0,
        MultiSSPropagation::Krylov,coarse,target) ||
       !PropagateOpticalScalar(file,options,0.05,2.0,
        MultiSSPropagation::Krylov,middle,target) ||
       !PropagateOpticalScalar(file,options,0.025,2.0,
        MultiSSPropagation::Krylov,fine,target))return false;
    const double ec=std::abs(coarse-exact),em=std::abs(middle-exact),ef=std::abs(fine-exact);
    return ef<em && em<ec && em/ec<0.35 && ef/em<0.35 && ef<4.0e-5;
}

bool test_multiss_experimental_fast_pulse_rk4_gate_and_krylov()
{
    OpticalTestFile file("time intensity\n0 0\n0.05 1\n0.1 0\n");
    auto A=ScalarSystem("A",true),B=ScalarSystem("B",false);
    A->Add(std::make_shared<SpinAPI::Transition>("pump",
        "type=sink;sourcestate=State;target=B;targetstate=State;"
        "rateprofile=experimental;"+file.FileProperties()+
        "intensityunit=relative;ratelaw=linear;ratescale=100;",A));
    std::vector<SpinAPI::system_ptr> systems{A,B};
    if(!ValidateSystemObjects(systems))return false;
    MultiSSExecutionPlan plan;plan.totalTime=0.2;plan.timeStep=0.05;
    plan.propagation=MultiSSPropagation::RK4;
    MultiSSOrientation o;SpinAPI::CreateZYZRotationMatrix(0,0,0,o.frameToLab);
    MultiSSNetwork network;std::string error;
    if(!MultiSSNetworkBuilder::Build(systems,plan,o,network,error))
    {std::cout<<error<<std::endl;return false;}
    MultiSSTrajectory trajectory;
    if(MultiSSPropagator::Propagate(plan,network,trajectory,error) ||
       error.find("RK4 step stability bound")==std::string::npos)
    {std::cout<<"missing fast optical RK4 stability error: "<<error<<std::endl;return false;}
    double source=0.0,target=0.0;
    if(!PropagateOpticalScalar(file,
        "intensityunit=relative;ratelaw=linear;ratescale=100;",
        0.005,0.2,MultiSSPropagation::Krylov,source,target))return false;
    return std::abs(source-std::exp(-5.0))<1.0e-9 &&
        std::abs(target-(1.0-std::exp(-5.0)))<1.0e-9;
}

void AddMultiSSGeneralTests(std::vector<test_case> &cases)
{
    cases.push_back(test_case("MultiSS Gaussian rate fraction normalization",test_multiss_timeprofile_gaussian_fraction_normalization));
    cases.push_back(test_case("MultiSS preservespins preserves nuclear coherence",test_multiss_preservespins_keeps_arbitrary_nuclear_coherence));
    cases.push_back(test_case("MultiSS constant transfer analytic kinetics",test_multiss_constant_transfer_matches_analytic_two_level_kinetics));
    cases.push_back(test_case("MultiSS Krylov fast transfer stability",test_multiss_krylov_handles_fast_transfer_without_rk4_stability_limit));
    cases.push_back(test_case("MultiSS exponential static plus event",test_multiss_exponential_allows_static_generator_with_exact_event_boundary));
    cases.push_back(test_case("MultiSS event boundary exact and once",test_multiss_instantaneous_event_lands_exactly_and_applies_once));
    cases.push_back(test_case("MultiSS timeintegrated solve semantics",test_multiss_timeintegrated_is_not_steadystate));
    cases.push_back(test_case("MultiSS true steady state",test_multiss_true_steadystate_closed_reversible_network));
    cases.push_back(test_case("MultiSS NZ core parity",test_multiss_nakajima_zwanzig_core_exact_algebra));
    cases.push_back(test_case("MultiSS parallel-channel rates add",test_multiss_three_parallel_channels_use_sum_of_rates));
    cases.push_back(test_case("MultiSS Mims push coherence map",test_multiss_mims_spin_selective_push_preserves_coherence_amplitude));
    cases.push_back(test_case("MultiSS timeintegrated matches long propagation",test_multiss_timeintegrated_matches_long_time_propagation));
    cases.push_back(test_case("MultiSS Gaussian transfer conserves trace",test_multiss_gaussian_transfer_conserves_trace_at_every_sample));
    cases.push_back(test_case("MultiSS trajectory profile interpolation and area",test_multiss_trajectory_timeprofile_interpolation_and_area));
    cases.push_back(test_case("MultiSS short Gaussian matches event",test_multiss_short_gaussian_fixed_area_matches_instantaneous_event));
    cases.push_back(test_case("MultiSS direct sum equals enlarged block sector",test_multiss_direct_sum_equals_enlarged_hilbert_block_diagonal_sector));
    cases.push_back(test_case("MultiSS common orientation and frame fallbacks",test_multiss_system_level_state_frames_use_common_orientation));
    cases.push_back(test_case("MultiSS experimental opt-in and named columns",test_multiss_experimental_opt_in_and_named_columns));
    cases.push_back(test_case("MultiSS experimental absorption CODATA and units",test_multiss_experimental_absorption_codata_and_units));
    cases.push_back(test_case("MultiSS experimental FileReader irradiance prefix case",test_multiss_experimental_file_reader_preserves_irradiance_prefix_case));
    cases.push_back(test_case("MultiSS experimental nonlinear intensity interpolation",test_multiss_experimental_nonlinear_interpolate_intensity_first));
    cases.push_back(test_case("MultiSS experimental malformed input rejection",test_multiss_experimental_malformed_input_rejected));
    cases.push_back(test_case("MultiSS experimental population and amplitude oracles",test_multiss_experimental_population_oracles_and_relative_amplitudes));
    cases.push_back(test_case("MultiSS experimental nonlinear population oracles",test_multiss_experimental_nonlinear_population_oracles));
    cases.push_back(test_case("MultiSS experimental shared waveform and cache rewrite",test_multiss_experimental_shared_waveform_distinct_rates_and_cache_rewrite));
    cases.push_back(test_case("MultiSS experimental edges event and time units",test_multiss_experimental_nonzero_edges_event_order_and_time_units));
    cases.push_back(test_case("MultiSS experimental diagnostics compatibility",test_multiss_experimental_diagnostics_and_legacy_layout));
    cases.push_back(test_case("MultiSS experimental preservespins coherence",test_multiss_experimental_preservespins_coherence));
    cases.push_back(test_case("MultiSS experimental sharp pulse convergence",test_multiss_experimental_sharp_pulse_timestep_convergence));
    cases.push_back(test_case("MultiSS experimental nonlinear Krylov convergence",test_multiss_experimental_nonlinear_krylov_timestep_convergence));
    cases.push_back(test_case("MultiSS experimental fast pulse stability gate",test_multiss_experimental_fast_pulse_rk4_gate_and_krylov));
}
