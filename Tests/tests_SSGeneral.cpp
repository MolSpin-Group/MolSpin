//////////////////////////////////////////////////////////////////////////////
// SSGeneral production hierarchy tests.
//////////////////////////////////////////////////////////////////////////////
#include "SSExecutionPlan.h"
#include "SSInteractionRelaxation.h"
#include "SSRedfieldBuilder.h"
#include "SSNakajimaZwanzigBuilder.h"
#include "NakajimaZwanzig.h"
#include "Redfield.h"
#include "Relaxation.h"
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
    std::string CorrelationMatrixText(std::size_t rows,
        const std::vector<std::vector<double>> &values)
    {
        std::ostringstream text;
        for(std::size_t row=0;row<rows;++row)
        {
            if(row>0)text<<",";
            text<<"[\"";
            const auto &entries=values[row];
            for(std::size_t column=0;column<entries.size();++column)
            {
                if(column>0)text<<" ";
                text<<entries[column];
            }
            text<<"\"]";
        }
        return text.str();
    }

    std::string CartesianMatrixCorrelationProperties(std::size_t activeRow,
        const std::vector<double> &amplitudes,
        const std::vector<double> &tauC,int terms,int spectralDensity=0)
    {
        const std::size_t rows=terms==1?9:81;
        std::vector<std::vector<double>> g(rows,
            std::vector<double>(amplitudes.size(),0.0));
        std::vector<std::vector<double>> tau(rows,tauC);
        g[activeRow]=amplitudes;
        return "ops=1;terms="+std::to_string(terms)+";def_multexpo=1;"
            "def_specdens="+std::to_string(spectralDensity)+";g="+
            CorrelationMatrixText(rows,g)+";tau_c="+
            CorrelationMatrixText(rows,tau)+";";
    }

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

    SpinAPI::system_ptr BuildSSNZLegacyParitySystem()
    {
        auto e=std::make_shared<SpinAPI::Spin>("E","type=electron;spin=1/2;tensor=isotropic(2.0023);");
        auto up=std::make_shared<SpinAPI::State>("Up","spin(E)=|1/2>;");
        auto down=std::make_shared<SpinAPI::State>("Down","spin(E)=|-1/2>;");
        auto identity=std::make_shared<SpinAPI::State>("Identity","");
        auto system=std::make_shared<SpinAPI::SpinSystem>("NZSystem");
        system->Add(e);system->Add(up);system->Add(down);system->Add(identity);
        const std::string correlation=CartesianMatrixCorrelationProperties(
            3,{-0.01,0.0},{0.2,0.0},0);
        system->Add(std::make_shared<SpinAPI::Interaction>("B0",
            "type=zeeman;field=\"0.0007 -0.0004 0.001\";spins=E;"+correlation));
        system->Add(std::make_shared<SpinAPI::Transition>("sink",
            "type=sink;sourcestate=Identity;rate=0.2;",system));
        system->SetProperties(std::make_shared<MSDParser::ObjectParser>(
            "properties","initialstate=Up;initialstatecoherences=keep;"));
        if(!up->ParseFromSystem(*system)||!down->ParseFromSystem(*system)||
           !identity->ParseFromSystem(*system)||!system->ValidateInteractions().empty())return nullptr;
        const std::vector<SpinAPI::system_ptr> systems{system};
        if(!system->ValidateTransitions(systems).empty())return nullptr;
        return system;
    }

    // High-field g-tensor-anisotropy limit from the supporting information of
    // JACS 2025, DOI 10.1021/jacs.5c06173 (Table S8 and eqs. S49-S51).
    // This intentionally omits hyperfine coupling, matching the fourth Se-dyad
    // row used to isolate the gta relaxation mechanism.
    SpinAPI::system_ptr BuildJacsSeHighFieldGtaSystem()
    {
        auto donor=std::make_shared<SpinAPI::Spin>("DonorElectron",
            "type=electron;spin=1/2;tensor=isotropic(2.0033);");
        auto acceptor=std::make_shared<SpinAPI::Spin>("AcceptorElectron",
            "type=electron;spin=1/2;tensor=isotropic(2.0033);");
        auto singlet=std::make_shared<SpinAPI::State>("Singlet",
            "spins(DonorElectron,AcceptorElectron)=|1/2,-1/2>-|-1/2,1/2>;");
        auto t0=std::make_shared<SpinAPI::State>("T0",
            "spins(DonorElectron,AcceptorElectron)=|1/2,-1/2>+|-1/2,1/2>;");
        auto tp=std::make_shared<SpinAPI::State>("Tp",
            "spin(DonorElectron)=|1/2>;spin(AcceptorElectron)=|1/2>;");
        auto tm=std::make_shared<SpinAPI::State>("Tm",
            "spin(DonorElectron)=|-1/2>;spin(AcceptorElectron)=|-1/2>;");
        auto identity=std::make_shared<SpinAPI::State>("Identity","");
        auto system=std::make_shared<SpinAPI::SpinSystem>("SeDyadHighField");
        system->Add(donor);system->Add(acceptor);system->Add(singlet);
        system->Add(t0);system->Add(tp);system->Add(tm);system->Add(identity);
        system->Add(std::make_shared<SpinAPI::Interaction>("DonorZeeman",
            "type=zeeman;field=\"0 0 10\";spins=DonorElectron;"));
        system->Add(std::make_shared<SpinAPI::Interaction>("AcceptorZeemanGta",
            "type=zeeman;field=\"0 0 10\";spins=AcceptorElectron;"
            "tau_c=0.6;g=0,0.1253,0.1253,0.1253,0.1253,0.1253;"
            "ops=0;coeff=0;def_g=1;terms=1;"));
        system->Add(std::make_shared<SpinAPI::Interaction>("Exchange",
            "type=doublespin;group1=DonorElectron;group2=AcceptorElectron;"
            "tensor=isotropic(-0.0189);prefactor=2.0023;ignoretensors=true;"));
        system->SetProperties(std::make_shared<MSDParser::ObjectParser>(
            "properties","initialstate=Singlet;initialstatecoherences=keep;"));
        if(!singlet->ParseFromSystem(*system)||!t0->ParseFromSystem(*system)||
           !tp->ParseFromSystem(*system)||!tm->ParseFromSystem(*system)||
           !identity->ParseFromSystem(*system)||!system->ValidateInteractions().empty())return nullptr;
        return system;
    }

    // One field point from Example/co2_formation/co2_formation.msd.  Keeping
    // this realistic three-spin model in the regression suite checks the full
    // task path, including rank-2 Zeeman relaxation and five competing sinks.
    SpinAPI::system_ptr BuildCO2FormationSystem()
    {
        auto e1=std::make_shared<SpinAPI::Spin>("RPElectron1",
            "type=electron;spin=1/2;tensor=isotropic(2.0007);");
        auto e2=std::make_shared<SpinAPI::Spin>("RPElectron2",
            "type=electron;spin=1/2;tensor=isotropic(2.0114);");
        auto h=std::make_shared<SpinAPI::Spin>("H",
            "spin=1/2;tensor=isotropic(0.05074);");
        auto singlet=std::make_shared<SpinAPI::State>("Singlet",
            "spins(RPElectron1,RPElectron2)=|1/2,-1/2>-|-1/2,1/2>;");
        auto t0=std::make_shared<SpinAPI::State>("T0",
            "spins(RPElectron1,RPElectron2)=|1/2,-1/2>+|-1/2,1/2>;");
        auto tp=std::make_shared<SpinAPI::State>("Tp",
            "spin(RPElectron2)=|1/2>;spin(RPElectron1)=|1/2>;");
        auto tm=std::make_shared<SpinAPI::State>("Tm",
            "spin(RPElectron2)=|-1/2>;spin(RPElectron1)=|-1/2>;");
        auto identity=std::make_shared<SpinAPI::State>("Identity","");
        auto system=std::make_shared<SpinAPI::SpinSystem>("system1");
        system->Add(e1);system->Add(e2);system->Add(h);
        system->Add(singlet);system->Add(t0);system->Add(tp);system->Add(tm);system->Add(identity);
        system->Add(std::make_shared<SpinAPI::Interaction>("zeeman1",
            "type=zeeman;field=\"0 0 0.03\";group1=RPElectron1;tau_c=0.0089;"
            "g=0,0.162057936,0.162057936,0.162057936,0.162057936,0.162057936;"
            "ops=0;def_g=1;def_specdens=1;terms=1;coeff=0;"));
        system->Add(std::make_shared<SpinAPI::Interaction>("zeeman2",
            "type=zeeman;field=\"0 0 0.03\";spins=RPElectron2;"));
        system->Add(std::make_shared<SpinAPI::Interaction>("radical1hyperfine",
            "type=hyperfine;group1=RPElectron2;group2=H;"));
        const std::vector<std::pair<std::string,std::string>> sinks={
            {"Product1","sourcestate=Singlet;rate=0.01;"},
            {"Product3","sourcestate=Singlet;rate=0.00001;"},
            {"Product4","sourcestate=T0;rate=0.00001;"},
            {"Product5","sourcestate=Tp;rate=0.00001;"},
            {"Product6","sourcestate=Tm;rate=0.00001;"}};
        for(const auto &sink:sinks)
            system->Add(std::make_shared<SpinAPI::Transition>(sink.first,
                "type=sink;"+sink.second,system));
        system->SetProperties(std::make_shared<MSDParser::ObjectParser>(
            "properties","initialstate=Identity;"));
        if(!singlet->ParseFromSystem(*system)||!t0->ParseFromSystem(*system)||
           !tp->ParseFromSystem(*system)||!tm->ParseFromSystem(*system)||
           !identity->ParseFromSystem(*system)||!system->ValidateInteractions().empty())return nullptr;
        const std::vector<SpinAPI::system_ptr> systems{system};
        if(!system->ValidateTransitions(systems).empty())return nullptr;
        return system;
    }

    bool RunSSNZTask(const SpinAPI::system_ptr &system,const std::string &properties,
        std::string &data)
    {
        RunSection::RunSection runSection;runSection.Add(system);
        MSDParser::ObjectParser parser("task",properties);
        runSection.Add(MSDParser::ObjectType::Task,parser);
        auto task=runSection.GetTask("task");if(!task)return false;
        std::ostringstream log,stream;task->SetLogStream(log);task->SetDataStream(stream);
        if(!runSection.Run(1))return false;
        data=stream.str();return true;
    }

    bool LastNumericRow(const std::string &data,std::vector<double> &row)
    {
        row.clear();std::istringstream lines(data);std::string line;
        while(std::getline(lines,line))
        {
            std::istringstream values(line);std::vector<double> candidate;double value=0.0;
            while(values>>value)candidate.push_back(value);
            if(!candidate.empty())row=std::move(candidate);
        }
        return !row.empty();
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

bool test_ssgeneral_explicit_orientation_rotates_full_hamiltonian_and_initial_state()
{
    MSDParser::ObjectParser parser("task",
        "type=SSGeneral;powderorientation=0.31 0.72 -0.19;");
    RunSection::General::SS::SSExecutionPlan plan;std::string error;
    if(!RunSection::General::SS::ResolveSSExecutionPlan(parser,plan,error)||
       plan.orientation!=RunSection::General::SS::SSOrientationMode::Explicit||
       plan.hamiltonianMode!=RunSection::General::SS::SSHamiltonianMode::RotatedFull)
    {
        std::cout<<"SS explicit plan diagnostic: "<<error<<" orientation="
                 <<static_cast<int>(plan.orientation)<<" mode="
                 <<static_cast<int>(plan.hamiltonianMode)<<std::endl;
        return false;
    }
    MSDParser::ObjectParser multiParser("task",
        "type=MultiSSGeneral;powderorientation=0.31 0.72 -0.19;");
    RunSection::General::MultiSS::MultiSSExecutionPlan multiPlan;
    if(!RunSection::General::MultiSS::ResolveMultiSSExecutionPlan(
        multiParser,multiPlan,error)||
       multiPlan.orientation!=RunSection::General::MultiSS::MultiSSOrientationMode::Explicit||
       multiPlan.hamiltonianMode!=RunSection::General::SS::SSHamiltonianMode::RotatedFull)
    {
        std::cout<<"MultiSS explicit plan diagnostic: "<<error<<" orientation="
                 <<static_cast<int>(multiPlan.orientation)<<" mode="
                 <<static_cast<int>(multiPlan.hamiltonianMode)<<std::endl;
        return false;
    }

    auto electron=std::make_shared<SpinAPI::Spin>("E","type=electron;spin=1/2;");
    auto up=std::make_shared<SpinAPI::State>("Up","spin(E)=|1/2>;");
    auto system=std::make_shared<SpinAPI::SpinSystem>("OrientedState");
    system->Add(electron);system->Add(up);
    system->SetProperties(std::make_shared<MSDParser::ObjectParser>("properties",
        "initialstate=Up;initialstateframe=molecular;"));
    if(!up->ParseFromSystem(*system)){std::cout<<"initial State parse failed"<<std::endl;return false;}

    SpinAPI::SpinSpace space(system);space.UseSuperoperatorSpace(false);
    arma::mat rotation;
    if(!SpinAPI::CreateZYZRotationMatrix(0.31,0.72,-0.19,rotation))return false;
    arma::cx_mat unrotated,expected,prepared;
    if(!space.GetState(up,unrotated)||!space.RotateState(unrotated,rotation,expected))
    {std::cout<<"initial State matrix preparation failed"<<std::endl;return false;}
    const arma::sp_cx_mat hamiltonian(2,2);
    if(!RunSection::General::SS::SSLiouvillianBuilder::BuildInitialDensity(
        system,space,hamiltonian,rotation,prepared,error))
    {std::cout<<"BuildInitialDensity failed: "<<error<<std::endl;return false;}
    const double expectedError=arma::norm(prepared-expected,"fro");
    const double rotationEffect=arma::norm(prepared-unrotated,"fro");
    if(!(expectedError<1.0e-12&&rotationEffect>1.0e-3))
        std::cout<<"initial frame diagnostic: expectedError="<<expectedError
                 <<" rotationEffect="<<rotationEffect<<std::endl;
    return expectedError<1.0e-12&&rotationEffect>1.0e-3;
}

bool test_ssgeneral_eigen_frame_is_orientation_specific_thermal_state()
{
    auto electron=std::make_shared<SpinAPI::Spin>("E","type=electron;spin=1/2;");
    auto thermalSystem=std::make_shared<SpinAPI::SpinSystem>("ThermalSystem");
    thermalSystem->Add(electron);
    thermalSystem->SetProperties(std::make_shared<MSDParser::ObjectParser>("properties",
        "initialstate=Thermal;initialstateframe=eigen;temperature=250;"));

    SpinAPI::SpinSpace thermalSpace(thermalSystem);
    arma::sp_cx_mat hamiltonian(2,2);
    hamiltonian(0,0)=arma::cx_double(-0.7,0.0);
    hamiltonian(1,1)=arma::cx_double(1.1,0.0);
    arma::cx_mat expected,prepared;
    std::string error;
    if(!thermalSpace.ThermalStateFromHamiltonian(
            arma::cx_mat(hamiltonian),250.0,expected)||
       !RunSection::General::SS::SSLiouvillianBuilder::BuildInitialDensity(
            thermalSystem,thermalSpace,hamiltonian,arma::eye<arma::mat>(3,3),prepared,error)||
       arma::norm(prepared-expected,"fro")>1.0e-13)return false;

    auto up=std::make_shared<SpinAPI::State>("Up","spin(E)=|1/2>;");
    auto namedSystem=std::make_shared<SpinAPI::SpinSystem>("NamedEigenSystem");
    namedSystem->Add(electron);namedSystem->Add(up);
    namedSystem->SetProperties(std::make_shared<MSDParser::ObjectParser>("properties",
        "initialstate=Up;initialstateframe=eigen;"));
    if(!up->ParseFromSystem(*namedSystem))return false;
    SpinAPI::SpinSpace namedSpace(namedSystem);
    return !RunSection::General::SS::SSLiouvillianBuilder::BuildInitialDensity(
        namedSystem,namedSpace,hamiltonian,arma::eye<arma::mat>(3,3),prepared,error)&&
        error.find("requires exactly one Thermal")!=std::string::npos;
}

bool test_ssgeneral_relaxation_operator_basis_uses_powder_and_interaction_frames()
{
    auto first=std::make_shared<SpinAPI::Spin>("E1","type=electron;spin=1/2;");
    auto second=std::make_shared<SpinAPI::Spin>("E2","type=electron;spin=1/2;");
    auto system=std::make_shared<SpinAPI::SpinSystem>("FrameSystem");
    system->Add(first);system->Add(second);
    auto interaction=std::make_shared<SpinAPI::Interaction>("DipolarNoise",
        "type=doublespin;group1=E1;group2=E2;tensor=isotropic(0.001);"
        "orientation=0.17,-0.29,0.41;");
    system->Add(interaction);
    const auto validation=system->ValidateInteractions();
    if(!validation.empty()){std::cout<<"frame interaction validation failed"<<std::endl;return false;}

    SpinAPI::SpinSpace space(system);space.UseSuperoperatorSpace(false);
    arma::mat powder,frame;
    if(!SpinAPI::CreateZYZRotationMatrix(-0.38,0.63,0.12,powder)||
       !SpinAPI::CreateZYZRotationMatrix(0.17,-0.29,0.41,frame))return false;
    std::vector<arma::cx_mat> operators;
    std::string error;
    if(!RunSection::General::SS::SSInteractionRelaxation::BuildOperatorBasis(
        space,interaction,first,second,1,powder,operators,error)||operators.size()!=9)
    {
        std::cout<<"frame operator build failed: "<<error<<" size="<<operators.size()<<std::endl;
        return false;
    }

    std::vector<arma::cx_mat> firstLab(3),secondLab(3);
    if(!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(first->Sx()),first,firstLab[0])||
       !space.CreateOperator(arma::conv_to<arma::cx_mat>::from(first->Sy()),first,firstLab[1])||
       !space.CreateOperator(arma::conv_to<arma::cx_mat>::from(first->Sz()),first,firstLab[2])||
       !space.CreateOperator(arma::conv_to<arma::cx_mat>::from(second->Sx()),second,secondLab[0])||
       !space.CreateOperator(arma::conv_to<arma::cx_mat>::from(second->Sy()),second,secondLab[1])||
       !space.CreateOperator(arma::conv_to<arma::cx_mat>::from(second->Sz()),second,secondLab[2]))return false;
    const arma::mat combined=powder*frame;
    for(std::size_t a=0;a<3;++a)for(std::size_t b=0;b<3;++b)
    {
        arma::cx_mat firstRot(arma::size(firstLab[0]),arma::fill::zeros);
        arma::cx_mat secondRot(arma::size(secondLab[0]),arma::fill::zeros);
        for(std::size_t lab=0;lab<3;++lab)
        {
            firstRot+=combined(lab,a)*firstLab[lab];
            secondRot+=combined(lab,b)*secondLab[lab];
        }
        const double mismatch=arma::norm(operators[3*a+b]-firstRot*secondRot,"fro");
        if(mismatch>1.0e-12)
        {std::cout<<"frame operator mismatch at "<<a<<","<<b<<": "<<mismatch<<std::endl;return false;}
    }
    return true;
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

bool test_ssgeneral_time_grid_has_single_exact_endpoint()
{
    auto system=BuildSSIdentityDecay(0.2);if(!system)return false;
    RunSection::General::SS::SSExecutionPlan plan;
    plan.calculation=RunSection::General::SS::SSCalculation::TimeEvolution;
    plan.propagation=RunSection::General::SS::SSPropagation::Exponential;
    plan.totalTime=500.0;plan.timeStep=0.4;
    RunSection::General::SS::SSOrientation orientation;
    RunSection::General::SS::SSPreparedCalculation prepared;std::string error;
    if(!RunSection::General::SS::SSSystemPreparation::Prepare(
        system,plan,orientation,prepared,error))return false;
    RunSection::General::SS::SSTrajectory trajectory;
    if(!RunSection::General::SS::SSPropagator::Propagate(
        plan,prepared,trajectory,error))return false;
    return trajectory.times.size()==1251&&trajectory.states.size()==1251&&
        trajectory.times.back()==500.0&&trajectory.times[1249]<500.0;
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



bool test_ssgeneral_nakajima_zwanzig_builder_exact_cartesian_parity()
{
    auto e=std::make_shared<SpinAPI::Spin>("E","type=electron;spin=1/2;tensor=isotropic(2.0023);");
    auto up=std::make_shared<SpinAPI::State>("Up","spin(E)=|1/2>;");
    auto system=std::make_shared<SpinAPI::SpinSystem>("NZSystem");
    system->Add(e);system->Add(up);
    // A tilted field makes the Hamiltonian eigenbasis different from the
    // propagation basis.  Unequal Cartesian amplitudes prevent rotational
    // invariance from hiding an omitted Liouville-space basis transform.
    auto z=std::make_shared<SpinAPI::Interaction>("B0",
        "type=zeeman;field=\"0.0007 -0.0004 0.001\";spins=E;ops=1;terms=1;"
        "def_g=1;g=0.2,0,0,0,0,0,0,0,0;tau_c=0.3;");
    system->Add(z);system->SetProperties(std::make_shared<MSDParser::ObjectParser>("properties","initialstate=Up;"));
    if(!up->ParseFromSystem(*system)||!system->ValidateInteractions().empty())return false;

    SpinAPI::SpinSpace space(system); arma::sp_cx_mat H;std::string error;
    if(!RunSection::General::SS::SSLiouvillianBuilder::BuildHamiltonian(system,space,
        RunSection::General::SS::SSHamiltonianMode::FixedFull,arma::eye<arma::mat>(3,3),H,error))return false;
    arma::sp_cx_mat R;
    if(!RunSection::General::SS::SSNakajimaZwanzigBuilder::Build(
        system,space,H,arma::eye<arma::mat>(3,3),R,error))return false;

    arma::vec eig;arma::cx_mat C;if(!arma::eig_sym(eig,C,arma::cx_mat(H)))return false;
    arma::cx_mat omega;if(!SpinAPI::NakajimaZwanzig::FrequencyMatrix(eig,omega,&error))return false;
    arma::cx_mat sx,sy,sz;
    if(!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(e->Sx()),e,sx)||
       !space.CreateOperator(arma::conv_to<arma::cx_mat>::from(e->Sy()),e,sy)||
       !space.CreateOperator(arma::conv_to<arma::cx_mat>::from(e->Sz()),e,sz))return false;
    std::vector<arma::cx_mat> ops={sx,sx,sx,sy,sy,sy,sz,sz,sz};
    const std::vector<double> amplitudes={0.2,0,0,0,0,0,0,0,0};
    arma::cx_mat expectedEigenbasis(4,4,arma::fill::zeros);
    for(size_t i=0;i<ops.size();++i)
    {
        ops[i]=C.t()*ops[i]*C;
        arma::cx_mat J,term;
        if(!SpinAPI::NakajimaZwanzig::ExponentialSpectralDensity(
            {amplitudes[i]*amplitudes[i],0.0},{0.3,0.0},omega,J,&error)||
           !SpinAPI::NakajimaZwanzig::RelaxationTensor(ops[i],ops[i],J,term,&error))return false;
        expectedEigenbasis+=term;
    }
    arma::sp_cx_mat expectedPropagationBasis;
    if(!space.TransformSuperoperatorFromEigenbasis(
        C,arma::sp_cx_mat(expectedEigenbasis),expectedPropagationBasis))return false;
    const arma::cx_mat got(R);
    const arma::cx_vec identity=arma::vectorise(arma::eye<arma::cx_mat>(2,2).t());
    const double parity=arma::norm(got-arma::cx_mat(expectedPropagationBasis),"fro");
    const double basisDifference=arma::norm(got-expectedEigenbasis,"fro");
    const double traceResidual=arma::norm(got*identity,2);
    if(!(parity<1e-12&&basisDifference>1e-6&&traceResidual<1e-12))
        std::cout<<"NZ basis diagnostic: parity="<<parity
                 <<" basisDifference="<<basisDifference
                 <<" traceResidual="<<traceResidual<<std::endl;
    return parity<1e-12&&basisDifference>1e-6&&traceResidual<1e-12;
}

bool test_ssgeneral_nakajima_zwanzig_plan_and_multiss_one_block_parity()
{
    MSDParser::ObjectParser p("task","type=SSGeneral;relaxationmodel=nakajimazwanzig;");
    RunSection::General::SS::SSExecutionPlan sp;std::string error;
    if(!RunSection::General::SS::ResolveSSExecutionPlan(p,sp,error)||sp.relaxationModel!=RunSection::General::SS::SSRelaxationModel::NakajimaZwanzig)return false;

    auto e=std::make_shared<SpinAPI::Spin>("E","type=electron;spin=1/2;tensor=isotropic(2.0023);");
    auto up=std::make_shared<SpinAPI::State>("Up","spin(E)=|1/2>;");
    auto system=std::make_shared<SpinAPI::SpinSystem>("NZSystem");system->Add(e);system->Add(up);
    auto z=std::make_shared<SpinAPI::Interaction>("B0","type=zeeman;field=\"0 0 0.001\";spins=E;ops=1;terms=1;g=0.2;tau_c=0.3;");
    system->Add(z);system->SetProperties(std::make_shared<MSDParser::ObjectParser>("properties","initialstate=Up;"));
    if(!up->ParseFromSystem(*system)||!system->ValidateInteractions().empty())return false;

    RunSection::General::SS::SSOrientation so;RunSection::General::SS::SSPreparedCalculation ss;
    if(!RunSection::General::SS::SSSystemPreparation::Prepare(system,sp,so,ss,error))return false;
    RunSection::General::MultiSS::MultiSSExecutionPlan mp;
    mp.relaxationModel=RunSection::General::SS::SSRelaxationModel::NakajimaZwanzig;
    RunSection::General::MultiSS::MultiSSOrientation mo;SpinAPI::CreateZYZRotationMatrix(0,0,0,mo.frameToLab);
    RunSection::General::MultiSS::MultiSSNetwork net;
    if(!RunSection::General::MultiSS::MultiSSNetworkBuilder::Build({system},mp,mo,net,error))return false;
    return arma::norm(arma::cx_mat(ss.generator)-arma::cx_mat(net.Generator(0.0)),"fro")<1e-12;
}

bool test_ssgeneral_nakajima_zwanzig_matches_legacy_task_observables()
{
    auto legacySystem=BuildSSNZLegacyParitySystem();
    auto generalSystem=BuildSSNZLegacyParitySystem();
    if(!legacySystem||!generalSystem)return false;

    std::string legacyData,generalData;
    if(!RunSSNZTask(legacySystem,
        "type=NakajimaZwanzig-Relaxation;transitionyields=false;",legacyData)||
       !RunSSNZTask(generalSystem,
        "type=SSGeneral;calculation=timeintegrated;observables=states;"
        "relaxationmodel=nakajima_zwanzig;hamiltonianmode=full;",generalData))return false;

    std::vector<double> legacy,general;
    if(!LastNumericRow(legacyData,legacy)||!LastNumericRow(generalData,general)||
       general.size()!=4||legacy.size()<general.size())return false;
    for(size_t i=0;i<general.size();++i)
    {
        const double scale=std::max({1.0,std::abs(legacy[i]),std::abs(general[i])});
        if(std::abs(legacy[i]-general[i])>2.0e-6*scale)return false;
    }
    return true;
}

bool test_ssgeneral_jacs_se_high_field_gta_benchmark()
{
    auto generalSystem=BuildJacsSeHighFieldGtaSystem();
    if(!generalSystem)return false;

    RunSection::General::SS::SSExecutionPlan plan;
    plan.calculation=RunSection::General::SS::SSCalculation::TimeEvolution;
    plan.propagation=RunSection::General::SS::SSPropagation::Exponential;
    plan.relaxationModel=RunSection::General::SS::SSRelaxationModel::NakajimaZwanzig;
    plan.totalTime=1.0;plan.timeStep=1.0;
    RunSection::General::SS::SSOrientation orientation;
    RunSection::General::SS::SSPreparedCalculation prepared;std::string error;
    if(!RunSection::General::SS::SSSystemPreparation::Prepare(
        generalSystem,plan,orientation,prepared,error))return false;
    RunSection::General::SS::SSTrajectory trajectory;
    if(!RunSection::General::SS::SSPropagator::Propagate(
        plan,prepared,trajectory,error)||trajectory.states.size()!=2)return false;
    RunSection::General::SS::SSObservableCollector collector;
    if(!collector.Prepare(plan,prepared,orientation,error))return false;
    arma::rowvec values;
    if(!collector.Evaluate(prepared,trajectory.states.back(),values,error)||values.n_elem!=5)return false;

    size_t t0Index=values.n_elem;
    const auto labels=collector.Labels();
    for(size_t i=0;i<labels.size();++i)
        if(labels[i].find(".T0.population")!=std::string::npos ||
           labels[i].find(".t0.population")!=std::string::npos)t0Index=i;
    if(t0Index>=values.n_elem)return false;

    // Eq. S51, using the rounded SI values: (g':g')=1.013e-5,
    // B=10 T, tau_c=0.6 ns and 2J=18.9 mT.  MolSpin's common
    // electron prefactor is gamma_e/g_e=87.9410005 rad ns^-1 T^-1.
    const double gammaElectron=2.0*87.9410005;
    const double exchangeJ=0.5*0.0189*gammaElectron;
    const double tau=0.6;
    const double kGta=(gammaElectron*gammaElectron/4.0)*100.0*1.013e-5*tau/
        (15.0*(1.0+4.0*exchangeJ*exchangeJ*tau*tau));
    const double expectedT0=0.5*(1.0-std::exp(-2.0*kGta)); // Eq. S49 at 1 ns
    return std::abs(values[t0Index]-expectedT0)<2.0e-5;
}

bool test_ssgeneral_co2_nz_legacy_and_multiss_task_parity()
{
    auto legacySystem=BuildCO2FormationSystem();
    auto generalSystem=BuildCO2FormationSystem();
    auto multiSystem=BuildCO2FormationSystem();
    if(!legacySystem||!generalSystem||!multiSystem)return false;

    std::string legacyData,generalData,multiData;
    if(!RunSSNZTask(legacySystem,
        "type=NakajimaZwanzig-Relaxation;transitionyields=true;",legacyData)||
       !RunSSNZTask(generalSystem,
        "type=SSGeneral;calculation=timeintegrated;observables=transitionyields;"
        "relaxationmodel=nakajima_zwanzig;hamiltonianmode=full;",generalData)||
       !RunSSNZTask(multiSystem,
        "type=MultiSSGeneral;calculation=timeintegrated;observables=states;"
        "transitionfluxes=true;relaxationmodel=nakajima_zwanzig;hamiltonianmode=full;",multiData))return false;

    std::vector<double> legacy,general,multi;
    if(!LastNumericRow(legacyData,legacy)||!LastNumericRow(generalData,general)||
       !LastNumericRow(multiData,multi)||legacy.size()!=7||general.size()!=6||multi.size()!=11)return false;
    double sum=0.0;
    for(size_t i=0;i<5;++i)
    {
        const double legacyYield=legacy[i+1];
        const double generalYield=general[i+1];
        const double multiFlux=multi[i+6];
        if(std::abs(generalYield-legacyYield)>6.0e-7||
           std::abs(generalYield-multiFlux)>1.0e-12)return false;
        sum+=generalYield;
    }
    return std::abs(general[1]-0.749406235791312)<1.0e-12&&
        std::abs(sum-1.0)<2.0e-12;
}

bool test_ssgeneral_nakajima_zwanzig_matrix_multiexponential_exact_parity()
{
    auto e=std::make_shared<SpinAPI::Spin>("E","type=electron;spin=1/2;tensor=isotropic(2.0023);");
    auto system=std::make_shared<SpinAPI::SpinSystem>("NZSystem");system->Add(e);
    // Row 3 is the ordered pair (Sx Bx, Sy Bx). Keeping only
    // this row nonzero makes the matrix-to-operator mapping observable.
    const std::string correlation=CartesianMatrixCorrelationProperties(
        3,{0.04,-0.01},{0.2,0.7},0);
    auto z=std::make_shared<SpinAPI::Interaction>("B0",
        "type=zeeman;field=\"0.0007 -0.0004 0.001\";spins=E;"+correlation);
    system->Add(z);if(!system->ValidateInteractions().empty())return false;
    SpinAPI::SpinSpace space(system);arma::sp_cx_mat H;std::string error;
    if(!RunSection::General::SS::SSLiouvillianBuilder::BuildHamiltonian(system,space,RunSection::General::SS::SSHamiltonianMode::FixedFull,arma::eye<arma::mat>(3,3),H,error))return false;
    arma::sp_cx_mat R;
    if(!RunSection::General::SS::SSNakajimaZwanzigBuilder::Build(
        system,space,H,arma::eye<arma::mat>(3,3),R,error))return false;

    arma::vec eigenvalues;arma::cx_mat eigenvectors;
    if(!arma::eig_sym(eigenvalues,eigenvectors,arma::cx_mat(H)))return false;
    arma::cx_mat omega;
    if(!SpinAPI::NakajimaZwanzig::FrequencyMatrix(eigenvalues,omega,&error))return false;
    arma::cx_mat sx,sy;
    if(!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(e->Sx()),e,sx)||
       !space.CreateOperator(arma::conv_to<arma::cx_mat>::from(e->Sy()),e,sy))return false;
    sx=eigenvectors.t()*sx*eigenvectors;
    sy=eigenvectors.t()*sy*eigenvectors;
    const std::vector<SpinAPI::Relaxation::ExponentialTerm> expansion={
        {0.04,0.2},{-0.01,0.7}};
    arma::cx_mat J,expectedEigenbasis;
    if(!SpinAPI::NakajimaZwanzig::SpectralDensity(
            expansion,omega,J,&error)||
       !SpinAPI::NakajimaZwanzig::RelaxationTensor(
            sx,sy,J,expectedEigenbasis,&error))return false;
    arma::sp_cx_mat expected;
    if(!space.TransformSuperoperatorFromEigenbasis(
            eigenvectors,arma::sp_cx_mat(expectedEigenbasis),expected))return false;
    return arma::norm(arma::cx_mat(R)-arma::cx_mat(expected),"fro")<1.0e-12;
}

bool test_ssgeneral_redfield_matrix_multiexponential_exact_parity()
{
    auto e=std::make_shared<SpinAPI::Spin>("E","type=electron;spin=1/2;tensor=isotropic(2.0023);");
    auto system=std::make_shared<SpinAPI::SpinSystem>("RedfieldSystem");system->Add(e);
    const std::string correlation=CartesianMatrixCorrelationProperties(
        3,{0.04,-0.01},{0.2,0.7},0,1);
    auto z=std::make_shared<SpinAPI::Interaction>("B0",
        "type=zeeman;field=\"0.0007 -0.0004 0.001\";spins=E;"+correlation);
    system->Add(z);if(!system->ValidateInteractions().empty())return false;
    SpinAPI::SpinSpace space(system);arma::sp_cx_mat H;std::string error;
    if(!RunSection::General::SS::SSLiouvillianBuilder::BuildHamiltonian(
        system,space,RunSection::General::SS::SSHamiltonianMode::FixedFull,
        arma::eye<arma::mat>(3,3),H,error))return false;
    arma::sp_cx_mat R;
    if(!RunSection::General::SS::SSRedfieldBuilder::Build(
        system,space,H,arma::eye<arma::mat>(3,3),R,error))return false;

    arma::vec eigenvalues;arma::cx_mat eigenvectors;
    if(!arma::eig_sym(eigenvalues,eigenvectors,arma::cx_mat(H)))return false;
    arma::cx_mat frequencies;
    if(!SpinAPI::Redfield::FrequencyMatrix(
        eigenvalues,frequencies,&error))return false;
    arma::cx_mat sx,sy;
    if(!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(e->Sx()),e,sx)||
       !space.CreateOperator(arma::conv_to<arma::cx_mat>::from(e->Sy()),e,sy))return false;
    sx=eigenvectors.t()*sx*eigenvectors;
    sy=eigenvectors.t()*sy*eigenvectors;
    const std::vector<SpinAPI::Relaxation::ExponentialTerm> expansion={
        {0.04,0.2},{-0.01,0.7}};
    arma::cx_mat J,first;
    if(!SpinAPI::Redfield::SpectralDensity(expansion,frequencies,
            SpinAPI::Relaxation::SpectralDensityFunction::RealLorentzian,J,&error)||
       !SpinAPI::Redfield::RelaxationTensor(sx,sy,J,first,&error))return false;
    arma::sp_cx_mat expected;
    if(!space.TransformSuperoperatorFromEigenbasis(
        eigenvectors,arma::sp_cx_mat(first),expected))return false;
    const arma::cx_rowvec trace=arma::vectorise(
        arma::eye<arma::cx_mat>(2,2).t()).t();
    return arma::norm(arma::cx_mat(R)-arma::cx_mat(expected),"fro")<1.0e-12&&
        arma::norm(trace*arma::cx_mat(R),2)<1.0e-12;
}

bool test_ssgeneral_redfield_plan_and_multiss_one_block_parity()
{
    MSDParser::ObjectParser ssParser("task","type=SSGeneral;relaxationmodel=redfield;");
    RunSection::General::SS::SSExecutionPlan ssPlan;std::string error;
    if(!RunSection::General::SS::ResolveSSExecutionPlan(ssParser,ssPlan,error)||
       ssPlan.relaxationModel!=RunSection::General::SS::SSRelaxationModel::Redfield)return false;
    MSDParser::ObjectParser multiParser("task","type=MultiSSGeneral;relaxationmodel=redfield;");
    RunSection::General::MultiSS::MultiSSExecutionPlan multiPlan;
    if(!RunSection::General::MultiSS::ResolveMultiSSExecutionPlan(
        multiParser,multiPlan,error)||
       multiPlan.relaxationModel!=RunSection::General::SS::SSRelaxationModel::Redfield)return false;

    auto e=std::make_shared<SpinAPI::Spin>("E","type=electron;spin=1/2;tensor=isotropic(2.0023);");
    auto up=std::make_shared<SpinAPI::State>("Up","spin(E)=|1/2>;");
    auto system=std::make_shared<SpinAPI::SpinSystem>("RedfieldSystem");
    system->Add(e);system->Add(up);
    system->Add(std::make_shared<SpinAPI::Interaction>("B0",
        "type=zeeman;field=\"0 0 0.001\";spins=E;ops=1;terms=1;"
        "def_g=1;g=0.2,0,0,0,0,0,0,0,0;tau_c=0.3;def_specdens=1;"));
    system->SetProperties(std::make_shared<MSDParser::ObjectParser>(
        "properties","initialstate=Up;"));
    if(!up->ParseFromSystem(*system)||!system->ValidateInteractions().empty())return false;

    RunSection::General::SS::SSOrientation ssOrientation;
    RunSection::General::SS::SSPreparedCalculation ssPrepared;
    if(!RunSection::General::SS::SSSystemPreparation::Prepare(
        system,ssPlan,ssOrientation,ssPrepared,error))return false;
    RunSection::General::MultiSS::MultiSSOrientation multiOrientation;
    SpinAPI::CreateZYZRotationMatrix(0,0,0,multiOrientation.frameToLab);
    RunSection::General::MultiSS::MultiSSNetwork network;
    if(!RunSection::General::MultiSS::MultiSSNetworkBuilder::Build(
        {system},multiPlan,multiOrientation,network,error))return false;
    return arma::norm(arma::cx_mat(ssPrepared.generator)-
        arma::cx_mat(network.Generator(0.0)),"fro")<1.0e-12;
}

bool test_ssgeneral_redfield_matches_legacy_task_observables()
{
    const auto build=[]()->SpinAPI::system_ptr
    {
        auto e=std::make_shared<SpinAPI::Spin>("E",
            "type=electron;spin=1/2;tensor=isotropic(2.0023);");
        auto up=std::make_shared<SpinAPI::State>("Up","spin(E)=|1/2>;");
        auto down=std::make_shared<SpinAPI::State>("Down","spin(E)=|-1/2>;");
        auto identity=std::make_shared<SpinAPI::State>("Identity","");
        auto system=std::make_shared<SpinAPI::SpinSystem>("RedfieldSystem");
        system->Add(e);system->Add(up);system->Add(down);system->Add(identity);
        // The negative term followed by zero padding catches two legacy edge
        // cases: signed-row detection and division by a padded tau_c=0.
        const std::string correlation=CartesianMatrixCorrelationProperties(
            3,{-0.01,0.0},{0.2,0.0},0,1);
        system->Add(std::make_shared<SpinAPI::Interaction>("B0",
            "type=zeeman;field=\"0.0007 -0.0004 0.001\";spins=E;"+correlation));
        system->Add(std::make_shared<SpinAPI::Transition>("sink",
            "type=sink;sourcestate=Identity;rate=0.2;",system));
        system->SetProperties(std::make_shared<MSDParser::ObjectParser>(
            "properties","initialstate=Up;initialstatecoherences=keep;"));
        if(!up->ParseFromSystem(*system)||!down->ParseFromSystem(*system)||
           !identity->ParseFromSystem(*system)||!system->ValidateInteractions().empty())return nullptr;
        const std::vector<SpinAPI::system_ptr> systems={system};
        if(!system->ValidateTransitions(systems).empty())return nullptr;
        return system;
    };
    auto legacySystem=build();auto generalSystem=build();
    if(!legacySystem||!generalSystem)return false;
    std::string legacyData,generalData;
    if(!RunSSNZTask(legacySystem,
        "type=redfield-relaxation;transitionyields=false;",legacyData)||
       !RunSSNZTask(generalSystem,
        "type=SSGeneral;calculation=timeintegrated;observables=states;"
        "relaxationmodel=redfield;hamiltonianmode=full;",generalData))return false;
    std::vector<double> legacy,general;
    if(!LastNumericRow(legacyData,legacy)||!LastNumericRow(generalData,general)||
       legacy.size()<general.size()||general.size()!=4)return false;
    for(std::size_t index=0;index<general.size();++index)
    {
        const double scale=std::max({1.0,std::abs(legacy[index]),std::abs(general[index])});
        if(std::abs(legacy[index]-general[index])>2.0e-6*scale)return false;
    }
    return true;
}

bool test_ssgeneral_matrix_correlation_validation()
{
    arma::mat amplitudes(2,2,arma::fill::ones);
    arma::mat tau(2,2,arma::fill::ones);
    SpinAPI::Relaxation::CorrelationExpansion expansion;std::string error;
    if(SpinAPI::Relaxation::CorrelationExpansion::PerChannel(
        9,true,amplitudes,tau,expansion,&error)||error.find("row count")==std::string::npos)return false;
    amplitudes.set_size(9,1);amplitudes.ones();tau.set_size(9,1);tau.ones();tau(4,0)=-1.0;
    if(SpinAPI::Relaxation::CorrelationExpansion::PerChannel(
        9,true,amplitudes,tau,expansion,&error)||error.find("tau_c > 0")==std::string::npos)return false;

    // Published fit matrices commonly pad shorter exponential expansions with
    // (g,tau_c)=(0,0). A zero amplitude has no physical contribution and must
    // not trigger either a division by zero or an invalid-input rejection.
    amplitudes.zeros(9,2);tau.zeros(9,2);
    amplitudes(3,0)=0.04;tau(3,0)=0.2;
    if(!SpinAPI::Relaxation::CorrelationExpansion::PerChannel(
        9,true,amplitudes,tau,expansion,&error))return false;
    const std::vector<SpinAPI::Relaxation::ExponentialTerm>*paddingTerms=nullptr;
    if(!expansion.Terms(3,3,paddingTerms,&error)||!paddingTerms||
       paddingTerms->size()!=1||std::abs((*paddingTerms)[0].amplitude-0.04)>1.0e-15||
       std::abs((*paddingTerms)[0].tauC-0.2)>1.0e-15)return false;

    auto e=std::make_shared<SpinAPI::Spin>("E",
        "type=electron;spin=1/2;tensor=isotropic(2.0023);");
    auto system=std::make_shared<SpinAPI::SpinSystem>("InvalidFlags");
    system->Add(e);
    auto interaction=std::make_shared<SpinAPI::Interaction>("B0",
        "type=zeeman;field=\"0 0 0.001\";spins=E;ops=1;terms=1;"
        "def_multexpo=2;g=0.1;tau_c=0.2;");
    system->Add(interaction);
    SpinAPI::Relaxation::CorrelationExpansion parsed;
    return !RunSection::General::SS::SSInteractionRelaxation::BuildCorrelationExpansion(
        interaction,9,1,parsed,error)&&error.find("def_multexpo must be 0 or 1")!=std::string::npos;
}

void AddSSGeneralTests(std::vector<test_case>&cases)
{
    cases.push_back({"SSGeneral execution plan is explicit",test_ssgeneral_execution_plan_is_explicit});
    cases.push_back({"SSGeneral explicit orientation rotates initial state",test_ssgeneral_explicit_orientation_rotates_full_hamiltonian_and_initial_state});
    cases.push_back({"SSGeneral eigen frame is orientation-specific thermal",test_ssgeneral_eigen_frame_is_orientation_specific_thermal_state});
    cases.push_back({"SSGeneral relaxation basis composes molecular frames",test_ssgeneral_relaxation_operator_basis_uses_powder_and_interaction_frames});
    cases.push_back({"SSGeneral timeintegrated identity decay analytic",test_ssgeneral_timeintegrated_identity_decay_is_analytic});
    cases.push_back({"SSGeneral time grid has one exact endpoint",test_ssgeneral_time_grid_has_single_exact_endpoint});
    cases.push_back({"SSGeneral matches MultiSS one-block generator",test_ssgeneral_matches_multiss_one_block_generator});
    cases.push_back({"SSGeneral task registration and execution",test_ssgeneral_task_is_registered_and_runs});
    cases.push_back({"SSGeneral rejects intersystem transition",test_ssgeneral_rejects_intersystem_transition});
    cases.push_back({"SSGeneral NZ Cartesian exact parity",test_ssgeneral_nakajima_zwanzig_builder_exact_cartesian_parity});
    cases.push_back({"SSGeneral NZ MultiSS one-block parity",test_ssgeneral_nakajima_zwanzig_plan_and_multiss_one_block_parity});
    cases.push_back({"SSGeneral NZ matches legacy task observables",test_ssgeneral_nakajima_zwanzig_matches_legacy_task_observables});
    cases.push_back({"SSGeneral JACS Se high-field gta benchmark",test_ssgeneral_jacs_se_high_field_gta_benchmark});
    cases.push_back({"SSGeneral CO2 NZ legacy and MultiSS task parity",test_ssgeneral_co2_nz_legacy_and_multiss_task_parity});
    cases.push_back({"SSGeneral NZ matrix multiexponential parity",test_ssgeneral_nakajima_zwanzig_matrix_multiexponential_exact_parity});
    cases.push_back({"SSGeneral Redfield matrix multiexponential parity",test_ssgeneral_redfield_matrix_multiexponential_exact_parity});
    cases.push_back({"SSGeneral Redfield MultiSS one-block parity",test_ssgeneral_redfield_plan_and_multiss_one_block_parity});
    cases.push_back({"SSGeneral Redfield matches legacy task observables",test_ssgeneral_redfield_matches_legacy_task_observables});
    cases.push_back({"SSGeneral matrix correlation validation",test_ssgeneral_matrix_correlation_validation});
}
