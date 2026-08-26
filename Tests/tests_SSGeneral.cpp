//////////////////////////////////////////////////////////////////////////////
// SSGeneral production hierarchy tests.
//////////////////////////////////////////////////////////////////////////////
#include "SSExecutionPlan.h"
#include "SSNakajimaZwanzigBuilder.h"
#include "NakajimaZwanzig.h"
#include "SpinSpace.h"
#include "SSOrientationSampler.h"
#include "SSSystemPreparation.h"
#include "SSPropagator.h"
#include "SSObservableCollector.h"
#include "TaskSSGeneral.h"
#include "MultiSSExecutionPlan.h"
#include "MultiSSOrientationSampler.h"
#include "MultiSSNetworkBuilder.h"
#include "Interaction.h"
#include "ObjectParser.h"
#include "Spin.h"
#include "SpinSystem.h"
#include "State.h"
#include "Transition.h"
#include "RunSection.h"
#include <cmath>
#include <sstream>

namespace
{
    SpinAPI::system_ptr BuildSSIdentityDecay(double rate)
    {
        auto e=std::make_shared<SpinAPI::Spin>("E","type=electron;spin=1/2;");
        auto up=std::make_shared<SpinAPI::State>("Up","spin(E)=|1/2>;");
        auto all=std::make_shared<SpinAPI::State>("All","");
        auto system=std::make_shared<SpinAPI::SpinSystem>("System");
        system->Add(e);system->Add(up);system->Add(all);
        auto sink=std::make_shared<SpinAPI::Transition>("sink","type=sink;sourcestate=All;rate="+std::to_string(rate)+";",system);
        system->Add(sink);
        system->SetProperties(std::make_shared<MSDParser::ObjectParser>("properties","initialstate=Up;"));
        if(!up->ParseFromSystem(*system)||!all->ParseFromSystem(*system))return nullptr;
        std::vector<SpinAPI::system_ptr> systems{system};
        if(!system->ValidateTransitions(systems).empty())return nullptr;
        return system;
    }
}

bool test_ssgeneral_execution_plan_is_explicit()
{
    MSDParser::ObjectParser p("task","type=SSGeneral;calculation=timeevolution;propagationmethod=rk4;observables=both;approximation=secular;powdersamplingpoints=3;powdergammapoints=2;");
    RunSection::General::SS::SSExecutionPlan plan;std::string error;
    return RunSection::General::SS::ResolveSSExecutionPlan(p,plan,error)&&
        plan.calculation==RunSection::General::SS::SSCalculation::TimeEvolution&&
        plan.propagation==RunSection::General::SS::SSPropagation::RK4&&
        plan.observables==RunSection::General::SS::SSObservableMode::Both&&
        plan.hamiltonianMode==RunSection::General::SS::SSHamiltonianMode::RotatedSecular&&
        plan.orientation==RunSection::General::SS::SSOrientationMode::PowderSO3;
}

bool test_ssgeneral_timeintegrated_identity_decay_is_analytic()
{
    const double k=0.2;auto system=BuildSSIdentityDecay(k);if(!system)return false;
    RunSection::General::SS::SSExecutionPlan plan;plan.calculation=RunSection::General::SS::SSCalculation::TimeIntegrated;
    RunSection::General::SS::SSOrientation o;std::string error;RunSection::General::SS::SSPreparedCalculation prepared;
    if(!RunSection::General::SS::SSSystemPreparation::Prepare(system,plan,o,prepared,error))return false;
    arma::cx_vec integrated;if(!RunSection::General::SS::SSPropagator::SolveTimeIntegrated(plan,prepared,integrated,error))return false;
    RunSection::General::SS::SSObservableCollector collector;
    if(!collector.Prepare(plan,prepared,o,error))return false;
    arma::rowvec values;
    if(!collector.Evaluate(prepared,integrated,values,error))return false;
    // Up and All are both fully occupied for this pure initial state, integrated for 1/k.
    return values.n_elem==2&&std::abs(values(0)-1.0/k)<1e-10&&std::abs(values(1)-1.0/k)<1e-10;
}

bool test_ssgeneral_matches_multiss_one_block_generator()
{
    auto system=BuildSSIdentityDecay(0.17);if(!system)return false;std::string error;
    RunSection::General::SS::SSExecutionPlan ssplan;RunSection::General::SS::SSOrientation sso;
    RunSection::General::SS::SSPreparedCalculation ss;
    if(!RunSection::General::SS::SSSystemPreparation::Prepare(system,ssplan,sso,ss,error))return false;

    RunSection::General::MultiSS::MultiSSExecutionPlan mp;
    RunSection::General::MultiSS::MultiSSOrientation mo;
    SpinAPI::CreateZYZRotationMatrix(0,0,0,mo.frameToLab);
    RunSection::General::MultiSS::MultiSSNetwork network;
    if(!RunSection::General::MultiSS::MultiSSNetworkBuilder::Build({system},mp,mo,network,error))return false;
    const arma::cx_mat A(ss.generator),B(network.Generator(0.0));
    return A.n_rows==B.n_rows&&arma::norm(A-B,"fro")<1e-12&&
        arma::norm(ss.initialState-network.systems.initialState,2)<1e-12;
}

bool test_ssgeneral_task_is_registered_and_runs()
{
    auto system=BuildSSIdentityDecay(0.2);if(!system)return false;RunSection::RunSection rs;rs.Add(system);
    MSDParser::ObjectParser parser("general","type=SSGeneral;calculation=timeintegrated;observables=states;");
    rs.Add(MSDParser::ObjectType::Task,parser);auto task=rs.GetTask("general");if(!task)return false;
    if(dynamic_cast<RunSection::General::SS::TaskSSGeneral*>(task.get())==nullptr)return false;
    std::ostringstream log,data;task->SetLogStream(log);task->SetDataStream(data);
    if(!rs.Run(1))return false;
    return log.str().find("--- SSGeneral resolved calculation ---")!=std::string::npos&&
        data.str().find("System.Up.population")!=std::string::npos;
}

bool test_ssgeneral_rejects_intersystem_transition()
{
    auto source=BuildSSIdentityDecay(0.2);if(!source)return false;
    auto target=std::make_shared<SpinAPI::SpinSystem>("Target");auto level=std::make_shared<SpinAPI::Spin>("L","spin=0;");auto st=std::make_shared<SpinAPI::State>("T","spin(L)=|0>;");target->Add(level);target->Add(st);st->ParseFromSystem(*target);
    // Replace source transition by adding an explicit inter-system channel; preparation must reject it.
    auto transfer=std::make_shared<SpinAPI::Transition>("transfer","type=sink;sourcestate=Up;target=Target;targetstate=T;rate=0.1;",source);source->Add(transfer);
    std::vector<SpinAPI::system_ptr> systems{source,target};if(!source->ValidateTransitions(systems).empty())return false;
    RunSection::General::SS::SSExecutionPlan plan;RunSection::General::SS::SSOrientation o;RunSection::General::SS::SSPreparedCalculation prepared;std::string error;
    return !RunSection::General::SS::SSSystemPreparation::Prepare(source,plan,o,prepared,error)&&error.find("MultiSSGeneral")!=std::string::npos;
}



bool test_ssgeneral_historical_nz_builder_exact_cartesian_parity()
{
    auto e=std::make_shared<SpinAPI::Spin>("E","type=electron;spin=1/2;tensor=isotropic(2.0023);");
    auto up=std::make_shared<SpinAPI::State>("Up","spin(E)=|1/2>;");
    auto system=std::make_shared<SpinAPI::SpinSystem>("NZSystem");
    system->Add(e);system->Add(up);
    auto z=std::make_shared<SpinAPI::Interaction>("B0","type=zeeman;field=\"0 0 0.001\";spins=E;ops=1;terms=1;g=0.2;tau_c=0.3;");
    system->Add(z);system->SetProperties(std::make_shared<MSDParser::ObjectParser>("properties","initialstate=Up;"));
    if(!up->ParseFromSystem(*system)||!system->ValidateInteractions().empty())return false;

    SpinAPI::SpinSpace space(system); arma::sp_cx_mat H;std::string error;
    if(!RunSection::General::SS::SSLiouvillianBuilder::BuildHamiltonian(system,space,
        RunSection::General::SS::SSHamiltonianMode::FixedFull,arma::eye<arma::mat>(3,3),H,error))return false;
    arma::sp_cx_mat R;
    if(!RunSection::General::SS::SSNakajimaZwanzigBuilder::BuildHistorical(system,space,H,R,error))return false;

    arma::vec eig;arma::cx_mat C;if(!arma::eig_sym(eig,C,arma::cx_mat(H)))return false;
    arma::cx_mat omega;if(!SpinAPI::NakajimaZwanzig::HistoricalFrequencyMatrix(eig,omega,&error))return false;
    arma::cx_mat J;if(!SpinAPI::NakajimaZwanzig::HistoricalMultiExponentialSpectralDensity({0.2},{0.3},omega,J,&error))return false;
    arma::cx_mat sx,sy,sz;
    if(!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(e->Sx()),e,sx)||
       !space.CreateOperator(arma::conv_to<arma::cx_mat>::from(e->Sy()),e,sy)||
       !space.CreateOperator(arma::conv_to<arma::cx_mat>::from(e->Sz()),e,sz))return false;
    std::vector<arma::cx_mat> ops={sx,sx,sx,sy,sy,sy,sz,sz,sz};
    arma::cx_mat expected(4,4,arma::fill::zeros);
    for(auto &op:ops){op=C.t()*op*C;arma::cx_mat term;if(!SpinAPI::NakajimaZwanzig::HistoricalTensor(op,op,J,term,&error))return false;expected+=term;}
    const arma::cx_mat got(R);
    const arma::cx_vec identity=arma::vectorise(arma::eye<arma::cx_mat>(2,2).t());
    return arma::norm(got-expected,"fro")<1e-12 && arma::norm(got*identity,2)<1e-12;
}

bool test_ssgeneral_historical_nz_plan_and_multiss_one_block_parity()
{
    MSDParser::ObjectParser p("task","type=SSGeneral;relaxationmodel=nakajimazwanzig;");
    RunSection::General::SS::SSExecutionPlan sp;std::string error;
    if(!RunSection::General::SS::ResolveSSExecutionPlan(p,sp,error)||sp.relaxationModel!=RunSection::General::SS::SSRelaxationModel::HistoricalNZ)return false;

    auto e=std::make_shared<SpinAPI::Spin>("E","type=electron;spin=1/2;tensor=isotropic(2.0023);");
    auto up=std::make_shared<SpinAPI::State>("Up","spin(E)=|1/2>;");
    auto system=std::make_shared<SpinAPI::SpinSystem>("NZSystem");system->Add(e);system->Add(up);
    auto z=std::make_shared<SpinAPI::Interaction>("B0","type=zeeman;field=\"0 0 0.001\";spins=E;ops=1;terms=1;g=0.2;tau_c=0.3;");
    system->Add(z);system->SetProperties(std::make_shared<MSDParser::ObjectParser>("properties","initialstate=Up;"));
    if(!up->ParseFromSystem(*system)||!system->ValidateInteractions().empty())return false;

    RunSection::General::SS::SSOrientation so;RunSection::General::SS::SSPreparedCalculation ss;
    if(!RunSection::General::SS::SSSystemPreparation::Prepare(system,sp,so,ss,error))return false;
    RunSection::General::MultiSS::MultiSSExecutionPlan mp;mp.historicalNZ=true;
    RunSection::General::MultiSS::MultiSSOrientation mo;SpinAPI::CreateZYZRotationMatrix(0,0,0,mo.frameToLab);
    RunSection::General::MultiSS::MultiSSNetwork net;
    if(!RunSection::General::MultiSS::MultiSSNetworkBuilder::Build({system},mp,mo,net,error))return false;
    return arma::norm(arma::cx_mat(ss.generator)-arma::cx_mat(net.Generator(0.0)),"fro")<1e-12;
}

bool test_ssgeneral_historical_nz_rejects_unmigrated_matrix_correlation_input()
{
    auto e=std::make_shared<SpinAPI::Spin>("E","type=electron;spin=1/2;");
    auto system=std::make_shared<SpinAPI::SpinSystem>("NZSystem");system->Add(e);
    auto z=std::make_shared<SpinAPI::Interaction>("B0","type=zeeman;field=\"0 0 0.001\";spins=E;ops=1;terms=1;def_multexpo=1;g=\"1 2;3 4\";tau_c=\"1 1;1 1\";");
    system->Add(z);if(!system->ValidateInteractions().empty())return false;
    SpinAPI::SpinSpace space(system);arma::sp_cx_mat H;std::string error;
    if(!RunSection::General::SS::SSLiouvillianBuilder::BuildHamiltonian(system,space,RunSection::General::SS::SSHamiltonianMode::FixedFull,arma::eye<arma::mat>(3,3),H,error))return false;
    arma::sp_cx_mat R;
    return !RunSection::General::SS::SSNakajimaZwanzigBuilder::BuildHistorical(system,space,H,R,error)&&error.find("def_multexpo=1")!=std::string::npos;
}

void AddSSGeneralTests(std::vector<test_case>&cases)
{
    cases.push_back({"SSGeneral execution plan is explicit",test_ssgeneral_execution_plan_is_explicit});
    cases.push_back({"SSGeneral timeintegrated identity decay analytic",test_ssgeneral_timeintegrated_identity_decay_is_analytic});
    cases.push_back({"SSGeneral matches MultiSS one-block generator",test_ssgeneral_matches_multiss_one_block_generator});
    cases.push_back({"SSGeneral task registration and execution",test_ssgeneral_task_is_registered_and_runs});
    cases.push_back({"SSGeneral rejects intersystem transition",test_ssgeneral_rejects_intersystem_transition});
    cases.push_back({"SSGeneral historical NZ Cartesian exact parity",test_ssgeneral_historical_nz_builder_exact_cartesian_parity});
    cases.push_back({"SSGeneral historical NZ MultiSS one-block parity",test_ssgeneral_historical_nz_plan_and_multiss_one_block_parity});
    cases.push_back({"SSGeneral historical NZ matrix-input parity gate",test_ssgeneral_historical_nz_rejects_unmigrated_matrix_correlation_input});
}
