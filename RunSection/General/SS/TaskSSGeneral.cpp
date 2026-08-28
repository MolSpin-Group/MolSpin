/////////////////////////////////////////////////////////////////////////
// TaskSSGeneral implementation (RunSection::General::SS)
// ------------------
// StaticSS and StaticSSTimeEvo remain independent numerical references.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "TaskSSGeneral.h"
#include "SSObservableCollector.h"
#include "SSOrientationSampler.h"
#include "SSPropagator.h"
#include "SSSystemPreparation.h"
#include "../GeneralLog.h"
#include "Settings.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include <cmath>
#include <iomanip>

namespace RunSection::General::SS
{
    TaskSSGeneral::TaskSSGeneral(const MSDParser::ObjectParser&p,const ::RunSection::RunSection&r):BasicTask(p,r){}
    bool TaskSSGeneral::Validate()
    {
        std::string error;if(!ResolveSSExecutionPlan(*this->Properties(),plan,error)){this->Log()<<"ERROR: Invalid SSGeneral execution plan: "<<error<<"."<<std::endl;return false;}planResolved=true;
        const auto systems=this->SpinSystems();if(systems.size()!=1){this->Log()<<"ERROR: SSGeneral owns exactly one SpinSystem; coupled manifolds belong to MultiSSGeneral."<<std::endl;return false;}
        const auto system=systems.front();if(!system)return false;SpinAPI::SpinSpace space(system);
        if(space.HasTimedependentInteractions()||space.HasTimedependentTransitions()){this->Log()<<"ERROR: time-dependent Interaction/Transition objects are not yet qualified in SSGeneral; use a dedicated dynamic task or MultiSSGeneral kinetic rate profiles as appropriate."<<std::endl;return false;}
        std::vector<SSOrientation> orientations;if(!SSOrientationSampler::Build(plan,orientations,this->Log(),error)){this->Log()<<"ERROR: "<<error<<"."<<std::endl;return false;}
        SSPreparedCalculation probe;if(!SSSystemPreparation::Prepare(system,plan,orientations.front(),probe,error)){this->Log()<<"ERROR: "<<error<<"."<<std::endl;return false;}
        SSObservableCollector collector;if(!collector.Prepare(plan,probe,orientations.front(),error)){this->Log()<<"ERROR: "<<error<<"."<<std::endl;return false;}
        this->Log()<<"\n--- SSGeneral resolved calculation ---"<<std::endl;
        this->Log()<<"Task mode: calculation="<<ToString(plan.calculation)<<", propagation="<<ToString(plan.propagation)<<", observables="<<ToString(plan.observables)<<", orientation="<<ToString(plan.orientation)<<", relaxation="<<ToString(plan.relaxationModel)<<"."<<std::endl;
        this->Log()<<"Hamiltonian approximation: "<<(plan.hamiltonianMode==SSHamiltonianMode::RotatedSecular?"high-field secular":"full")<<"."<<std::endl;
        if(plan.hamiltonianMode==SSHamiltonianMode::RotatedSecular)this->Log()<<"    Note: static high-field secularization is not a rotating-wave approximation. RWA/frequency/selective-spin semantics belong to a configured drive Interaction/Pulse."<<std::endl;
        this->Log()<<"Liouville dimension="<<probe.generator.n_rows<<", reaction operator=Haberkorn."<<std::endl;
        ::RunSection::General::Log::PrintSystemInventory(this->Log(),systems,"SSGeneral physics objects");double sum=0;for(const auto&o:orientations)sum+=o.weight;::RunSection::General::Log::PrintPowderSummary(this->Log(),ToString(plan.orientation),orientations.size(),sum);return true;
    }
    bool TaskSSGeneral::WriteHeader(const std::vector<std::string>&labels,bool timeEvolution)
    {this->Data()<<"Step ";if(timeEvolution)this->Data()<<"Time(ns) ";this->WriteStandardOutputHeader(this->Data());for(const auto&l:labels)this->Data()<<l<<" ";this->Data()<<std::endl;return true;}
    bool TaskSSGeneral::RunLocal()
    {
        if(!planResolved){std::string e;if(!ResolveSSExecutionPlan(*this->Properties(),plan,e))return false;}
        const auto system=this->SpinSystems().front();std::string error;std::vector<SSOrientation> orientations;if(!SSOrientationSampler::Build(plan,orientations,this->Log(),error)){this->Log()<<"ERROR: "<<error<<"."<<std::endl;return false;}
        std::vector<std::string> labels;std::vector<double> times;std::vector<arma::rowvec> averagedTrajectory;arma::rowvec averagedStatic;bool first=true;
        for(size_t oi=0;oi<orientations.size();++oi)
        {
            ::RunSection::General::Log::PrintOrientationProgress(this->Log(),oi,orientations.size());SSPreparedCalculation prepared;if(!SSSystemPreparation::Prepare(system,plan,orientations[oi],prepared,error)){this->Log()<<"ERROR: orientation "<<oi<<": "<<error<<"."<<std::endl;return false;}
            SSObservableCollector collector;if(!collector.Prepare(plan,prepared,orientations[oi],error)){this->Log()<<"ERROR: "<<error<<"."<<std::endl;return false;}if(first)labels=collector.Labels();else if(labels!=collector.Labels()){this->Log()<<"ERROR: observable layout changed between orientations."<<std::endl;return false;}
            if(plan.calculation==SSCalculation::TimeEvolution)
            {
                SSTrajectory trajectory;if(!SSPropagator::Propagate(plan,prepared,trajectory,error)){this->Log()<<"ERROR: "<<error<<"."<<std::endl;return false;}
                if(first){times=trajectory.times;averagedTrajectory.assign(times.size(),arma::rowvec(labels.size(),arma::fill::zeros));}
                if(trajectory.times.size()!=times.size()){this->Log()<<"ERROR: time grid changed between orientations."<<std::endl;return false;}
                for(size_t ti=0;ti<times.size();++ti){arma::rowvec values;if(!collector.Evaluate(prepared,trajectory.states[ti],values,error)){this->Log()<<"ERROR: "<<error<<"."<<std::endl;return false;}averagedTrajectory[ti]+=orientations[oi].weight*values;}
            }
            else
            {
                arma::cx_vec state;const bool ok=plan.calculation==SSCalculation::TimeIntegrated?SSPropagator::SolveTimeIntegrated(plan,prepared,state,error):SSPropagator::SolveSteadyState(plan,prepared,state,error);if(!ok){this->Log()<<"ERROR: "<<error<<"."<<std::endl;return false;}arma::rowvec values;if(!collector.Evaluate(prepared,state,values,error)){this->Log()<<"ERROR: "<<error<<"."<<std::endl;return false;}if(first)averagedStatic.zeros(values.n_elem);averagedStatic+=orientations[oi].weight*values;
            }first=false;
        }
        if(this->RunSettings()->CurrentStep()==1)WriteHeader(labels,plan.calculation==SSCalculation::TimeEvolution);
        if(plan.calculation==SSCalculation::TimeEvolution)for(size_t i=0;i<times.size();++i){this->Data()<<this->RunSettings()->CurrentStep()<<" "<<std::setprecision(15)<<times[i]<<" ";this->WriteStandardOutput(this->Data());for(double v:averagedTrajectory[i])this->Data()<<v<<" ";this->Data()<<std::endl;}
        else{this->Data()<<this->RunSettings()->CurrentStep()<<" ";this->WriteStandardOutput(this->Data());for(double v:averagedStatic)this->Data()<<std::setprecision(15)<<v<<" ";this->Data()<<std::endl;}return true;
    }
}
