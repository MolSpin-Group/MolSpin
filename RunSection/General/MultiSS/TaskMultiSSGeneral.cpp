/////////////////////////////////////////////////////////////////////////
// TaskMultiSSGeneral implementation (RunSection::General::MultiSS)
// ----------------------------------------------------------------------
// THIS CLASS IS INTENTIONALLY THIN.
//   TaskMultiSSGeneral owns lifecycle, validation, orientation averaging and
//   output formatting only.  Physics is delegated down the hierarchy:
//
//       SpinAPI
//         TimeProfile / TransferChannel / QuantumMap / established NZ algebra
//             -> RunSection::General::SS
//                  one-manifold superspace physics
//             -> RunSection::General::MultiSS
//                  direct-sum graph / events / propagation / observables
//             -> TaskMultiSSGeneral                         [this file]
//
// LEGACY BOUNDARY
//   RunSection/Tasks/TaskMultiStaticSS*, task-local NZ and Redfield classes are
//   NOT backends of this task and are not modified here.  They remain runnable
//   reference implementations during migration.
//
// LITERATURE MAP FOR THE IMPLEMENTED PHYSICS
//   * kinetic S1/CSS hierarchy and preserved spin memory:
//       DOI: 10.1039/D6CP00916F
//   * finite Gaussian optical-rate precedent in solid-state/NV kinetics:
//       DOI: 10.1038/ncomms14000
//   * instantaneous spin-selective pump/push event limit:
//       DOI: 10.1126/science.abl4254
//   * population/delayed-fluorescence collaborator application:
//       DOI: 10.1039/D6SC02081J
//   * published MolSpin NZ validation target:
//       DOI: 10.1021/jacs.5c06173
//   * formal reactive NZ reference, kept distinct from published-model parity:
//       DOI: 10.1063/5.0040519
//
// CURRENT SCOPE
//   Incoherent optical kinetics are TransferChannel rate profiles or QuantumMap
//   events.  Coherent optical excitation and finite magnetic/RF/MW pulse
//   Hamiltonians require distinct explicit drive machinery and are not silently
//   approximated by population-transfer rates in this implementation.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// DEVELOPER WORKFLOW / OWNERSHIP MAP
// ----------------------------------------------------------------------
// Top-level MultiSSGeneral orchestration.
//
// What is done here:
//   - Resolves the network execution plan and common orientation ensemble.
//   - For each crystallite: prepares local manifolds, compiles the kinetic graph, propagates/solves, evaluates observables and powder-averages the result.
//   - Owns validation/log/output sequencing rather than local spin physics.
//
// Connections to the General framework / SpinAPI:
//   - Uses MultiSSExecutionPlan, MultiSSOrientationSampler, MultiSSNetworkBuilder, MultiSSPropagator and MultiSSObservableCollector.
//   - Reuses General/SS for local one-manifold Hamiltonian, state and relaxation construction.
//   - SpinAPI supplies physical SpinSystem primitives and TransferChannel/quantum-map machinery.
//
// Why this ownership is used:
//   - MultiSSGeneral should remain the network composition layer; local relaxation improvements belong in General/SS or SpinAPI.
//
// TODO:
//   - A shared strain/SW ensemble driver must provide the same realization to all rigidly related manifolds when the fluctuating parameter is physically common.
/////////////////////////////////////////////////////////////////////////

#include "TaskMultiSSGeneral.h"

#include "MultiSSNetworkBuilder.h"
#include "MultiSSObservableCollector.h"
#include "MultiSSOrientationSampler.h"
#include "MultiSSPropagator.h"
#include "../GeneralLog.h"
#include "Settings.h"
#include "SpinSpace.h"
#include "SpinSystem.h"

#include <cmath>
#include <iomanip>

namespace RunSection::General::MultiSS
{
    TaskMultiSSGeneral::TaskMultiSSGeneral(const MSDParser::ObjectParser &parser,
        const ::RunSection::RunSection &runsection):BasicTask(parser,runsection) {}

    bool TaskMultiSSGeneral::Validate()
    {
        std::string error;
        if(!ResolveMultiSSExecutionPlan(*this->Properties(),plan,error))
        {this->Log()<<"ERROR: "<<error<<"."<<std::endl;return false;}
        const auto systems=this->SpinSystems();
        if(systems.empty()){this->Log()<<"ERROR: MultiSSGeneral requires at least one SpinSystem."<<std::endl;return false;}
        for(const auto &system:systems)
        {
            if(system==nullptr)return false;
            SpinAPI::SpinSpace space(system);
            if(space.HasTimedependentInteractions())
            {
                this->Log()<<"ERROR: continuously time-dependent Interaction objects are not yet part of General/MultiSS; "
                    <<"finite optical kinetics must use Transition rateprofile, while magnetic-drive integration remains a separate event/Hamiltonian extension."<<std::endl;
                return false;
            }
        }

        std::vector<MultiSSOrientation> orientations;
        if(!MultiSSOrientationSampler::Build(plan,orientations,this->Log(),error))
        {this->Log()<<"ERROR: "<<error<<"."<<std::endl;return false;}
        MultiSSNetwork probe;
        if(!MultiSSNetworkBuilder::Build(systems,plan,orientations.front(),probe,error))
        {this->Log()<<"ERROR: "<<error<<"."<<std::endl;return false;}

        this->Log()<<"\n--- MultiSSGeneral resolved calculation ---"<<std::endl;
        this->Log()<<"Task mode: calculation="<<ToString(plan.calculation)
            <<", propagation="<<ToString(plan.propagation)
            <<", observables="<<ToString(plan.observables)
            <<", orientation="<<ToString(plan.orientation)<<"."<<std::endl;
        this->Log()<<"Hamiltonian approximation: "
            <<(plan.hamiltonianMode==::RunSection::General::SS::SSHamiltonianMode::RotatedSecular
                ?"high-field secular":"full")<<"."<<std::endl;
        if(plan.hamiltonianMode==::RunSection::General::SS::SSHamiltonianMode::RotatedSecular)
            this->Log()<<"    Note: static high-field secularization is not a rotating-wave approximation. "
                <<"RWA/frequency/selective-spin handling belongs to the configured drive Interaction/Pulse."<<std::endl;
        this->Log()<<"Direct-sum Liouville dimension="<<probe.systems.globalDimension
            <<", continuous transfer channels="<<probe.continuousChannels.size()
            <<", instantaneous events="<<probe.events.size()<<"."<<std::endl;
        ::RunSection::General::Log::PrintSystemInventory(this->Log(),systems,"MultiSSGeneral physics objects");
        double weightSum=0.0; for(const auto &orientation:orientations) weightSum+=orientation.weight;
        ::RunSection::General::Log::PrintPowderSummary(this->Log(),ToString(plan.orientation),orientations.size(),weightSum);
        return true;
    }

    bool TaskMultiSSGeneral::WriteHeader(const std::vector<std::string> &labels,bool timeEvolution)
    {
        this->Data()<<"Step ";
        this->WriteStandardOutputHeader(this->Data());
        if(timeEvolution)this->Data()<<"EvolutionTime ";
        for(const auto &label:labels)this->Data()<<label<<" ";
        this->Data()<<std::endl;return true;
    }

    bool TaskMultiSSGeneral::RunLocal()
    {
        const auto systems=this->SpinSystems();
        std::string error;
        std::vector<MultiSSOrientation> orientations;
        if(!MultiSSOrientationSampler::Build(plan,orientations,this->Log(),error))
        {this->Log()<<"ERROR: "<<error<<"."<<std::endl;return false;}

        std::vector<std::string> labels;
        std::vector<double> referenceTimes;
        std::vector<arma::rowvec> averagedTrajectory;
        arma::rowvec averagedStatic;
        bool first=true;

        for(size_t oi=0;oi<orientations.size();++oi)
        {
            ::RunSection::General::Log::PrintOrientationProgress(this->Log(),oi,orientations.size());
            MultiSSNetwork network;
            if(!MultiSSNetworkBuilder::Build(systems,plan,orientations[oi],network,error))
            {this->Log()<<"ERROR: orientation "<<oi<<": "<<error<<"."<<std::endl;return false;}
            MultiSSObservableCollector collector;
            if(!collector.Prepare(plan,network,orientations[oi],error))
            {this->Log()<<"ERROR: "<<error<<"."<<std::endl;return false;}
            if(first)labels=collector.Labels();
            else if(labels!=collector.Labels())
            {this->Log()<<"ERROR: observable layout changed between orientations."<<std::endl;return false;}

            if(plan.calculation==MultiSSCalculation::TimeEvolution)
            {
                MultiSSTrajectory trajectory;
                if(!MultiSSPropagator::Propagate(plan,network,trajectory,error))
                {this->Log()<<"ERROR: "<<error<<"."<<std::endl;return false;}
                if(first)
                {
                    referenceTimes=trajectory.times;
                    averagedTrajectory.assign(referenceTimes.size(),arma::rowvec(labels.size(),arma::fill::zeros));
                }
                if(trajectory.times.size()!=referenceTimes.size())
                {this->Log()<<"ERROR: time grid changed between orientations."<<std::endl;return false;}
                for(size_t ti=0;ti<trajectory.times.size();++ti)
                {
                    if(std::abs(trajectory.times[ti]-referenceTimes[ti])>1.0e-10*std::max(1.0,std::abs(referenceTimes[ti])))
                    {this->Log()<<"ERROR: event-aware time grid is not common across orientations."<<std::endl;return false;}
                    arma::rowvec values;
                    if(!collector.Evaluate(network,trajectory.states[ti],trajectory.times[ti],values,error))
                    {this->Log()<<"ERROR: "<<error<<"."<<std::endl;return false;}
                    averagedTrajectory[ti]+=orientations[oi].weight*values;
                }
            }
            else
            {
                arma::cx_vec state;
                const bool ok=plan.calculation==MultiSSCalculation::TimeIntegrated
                    ?MultiSSPropagator::SolveTimeIntegrated(plan,network,state,error)
                    :MultiSSPropagator::SolveSteadyState(plan,network,state,error);
                if(!ok){this->Log()<<"ERROR: "<<error<<"."<<std::endl;return false;}
                arma::rowvec values;
                if(!collector.Evaluate(network,state,0.0,values,error))
                {this->Log()<<"ERROR: "<<error<<"."<<std::endl;return false;}
                if(first)averagedStatic.zeros(values.n_elem);
                averagedStatic+=orientations[oi].weight*values;
            }
            first=false;
        }

        if(this->RunSettings()->CurrentStep()==1)
            WriteHeader(labels,plan.calculation==MultiSSCalculation::TimeEvolution);

        if(plan.calculation==MultiSSCalculation::TimeEvolution)
        {
            for(size_t i=0;i<referenceTimes.size();++i)
            {
                this->Data()<<this->RunSettings()->CurrentStep()<<" ";
                this->WriteStandardOutput(this->Data());
                this->Data()<<std::setprecision(15)<<referenceTimes[i]<<" ";
                for(double v:averagedTrajectory[i])this->Data()<<v<<" ";
                this->Data()<<std::endl;
            }
        }
        else
        {
            this->Data()<<this->RunSettings()->CurrentStep()<<" ";
            this->WriteStandardOutput(this->Data());
            for(double v:averagedStatic)this->Data()<<std::setprecision(15)<<v<<" ";
            this->Data()<<std::endl;
        }
        return true;
    }
}
