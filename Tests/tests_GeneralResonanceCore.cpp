//////////////////////////////////////////////////////////////////////////////
// Reusable General resonance-core tests.
//
// The production resonance code under test is backend-neutral and has no
// BasicTask ownership. Frozen StaticHS-Resonance-Spectra is used only here as
// an external numerical oracle for compact one-orientation field sweeps.
//////////////////////////////////////////////////////////////////////////////
#include "GeneralResonanceHamiltonian.h"
#include "ExactResonanceSolver.h"
#include "HybridNuclearResonanceSolver.h"
#include "HybridNuclearResonancePreparation.h"
#include "ResonanceFieldJacobian.h"
#include "ResonanceLineshape.h"
#include "ResonanceMagneticMomentBuilder.h"
#include "ResonanceSpectrumEvaluator.h"
#include "ResonanceTransitionDetector.h"
#include "ResonanceTransitionMoments.h"
#include "HSHamiltonianBuilder.h"
#include "Interaction.h"
#include "NuclearZeeman.h"
#include "ObjectParser.h"
#include "RunSection.h"
#include "Spin.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "State.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    constexpr double GRC_MU_B_OVER_HBAR = 8.79410005e+1;

    struct GRC_Model
    {
        double gx = 2.0023;
        double gy = 2.0023;
        double gz = 2.0023;
        double hyperfine = 0.0; // rad/ns, isotropic electron-nucleus coupling
    };

    SpinAPI::system_ptr GRC_BuildSystem(const GRC_Model &model, double fieldT)
    {
        std::ostringstream eprops;
        eprops << std::setprecision(17)
               << "type=electron;spin=1/2;tensor=anisotropic("
               << model.gx << " " << model.gy << " " << model.gz << ");";
        auto electron = std::make_shared<SpinAPI::Spin>("E", eprops.str());

        std::ostringstream bprops;
        bprops << std::setprecision(17)
               << "type=zeeman;spins=E;field=0 0 " << fieldT
               << ";ignoretensors=false;commonprefactor=true;prefactor=1.0;";
        auto field = std::make_shared<SpinAPI::Interaction>("B0", bprops.str());
        auto up = std::make_shared<SpinAPI::State>("Up", "spin(E)=|1/2>;");

        auto system = std::make_shared<SpinAPI::SpinSystem>("System");
        system->Add(electron);
        system->Add(field);
        system->Add(up);

        if (model.hyperfine != 0.0)
        {
            auto nucleus = std::make_shared<SpinAPI::Spin>("N", "type=nucleus;spin=1/2;tensor=isotropic(1);");
            std::ostringstream aprops;
            aprops << std::setprecision(17)
                   << "type=hyperfine;group1=E;group2=N;tensor=isotropic("
                   << model.hyperfine
                   << ");ignoretensors=true;commonprefactor=false;prefactor=1.0;";
            auto hfc = std::make_shared<SpinAPI::Interaction>("A", aprops.str());
            system->Add(nucleus);
            system->Add(hfc);
        }

        system->SetProperties(std::make_shared<MSDParser::ObjectParser>("properties", "initialstate=Up;"));
        if (!system->ValidateInteractions().empty()) return nullptr;
        if (!up->ParseFromSystem(*system)) return nullptr;
        return system;
    }

    RunSection::General::HS::HSExecutionPlan GRC_Plan(const GRC_Model &model, bool secular=false)
    {
        RunSection::General::HS::HSExecutionPlan plan;
        plan.dynamics = RunSection::General::HS::Dynamics::Static;
        plan.calculation = RunSection::General::HS::Calculation::TimeEvolution;
        plan.sampling = RunSection::General::HS::Sampling::Direct;
        plan.orientation = RunSection::General::HS::OrientationMode::Explicit;
        plan.approximation = secular ? SpinAPI::HamiltonianApproximation::Secular
                                     : SpinAPI::HamiltonianApproximation::Full;
        plan.hasH0List = true;
        plan.h0List = {"B0"};
        if (model.hyperfine != 0.0) plan.h0List.push_back("A");
        return plan;
    }

    RunSection::General::HS::HSOrientation GRC_IdentityOrientation()
    {
        RunSection::General::HS::HSOrientation orientation;
        orientation.weight = 1.0;
        orientation.frameToLab.eye(3,3);
        return orientation;
    }

    bool GRC_GenerateFirstOrder(
        const arma::sp_cx_mat &coreHamiltonian,
        const arma::cx_mat &coreDensity,
        const arma::sp_cx_mat &coreDHdB,
        const arma::cx_mat &coreMuX,
        const arma::cx_mat &coreMuY,
        const RunSection::General::Resonance::
            HybridNuclearResonanceNucleus &nucleus,
        const RunSection::General::Resonance::
            SpectrumRequest &request,
        RunSection::General::Resonance::
            ResonanceLineSet &lines,
        std::string &error)
    {
        using namespace RunSection::General::Resonance;

        HybridNuclearResonancePoint point;
        point.coreHamiltonian=coreHamiltonian;
        point.coreDensity=coreDensity;
        point.coreDHdB=coreDHdB;
        point.coreMuX=coreMuX;
        point.coreMuY=coreMuY;
        point.hybrid.nuclei={nucleus};

        HybridNuclearResonanceReport report;
        return HybridNuclearResonanceSolver::GenerateFirstOrder(
            point,request,lines,report,error);
    }

    bool GRC_GenerateFirstOrder(
        const arma::sp_cx_mat &coreHamiltonian,
        const arma::cx_mat &coreDensity,
        const arma::sp_cx_mat &coreDHdB,
        const arma::cx_mat &coreMuX,
        const arma::cx_mat &coreMuY,
        const RunSection::General::Resonance::
            HybridNuclearResonanceRequest &hybrid,
        const RunSection::General::Resonance::
            SpectrumRequest &request,
        RunSection::General::Resonance::
            ResonanceLineSet &lines,
        RunSection::General::Resonance::
            HybridNuclearResonanceReport &report,
        std::string &error)
    {
        using namespace RunSection::General::Resonance;

        HybridNuclearResonancePoint point;
        point.coreHamiltonian=coreHamiltonian;
        point.coreDensity=coreDensity;
        point.coreDHdB=coreDHdB;
        point.coreMuX=coreMuX;
        point.coreMuY=coreMuY;
        point.hybrid=hybrid;

        return HybridNuclearResonanceSolver::GenerateFirstOrder(
            point,request,lines,report,error);
    }

    bool GRC_GenerateFirstOrderFiniteDifference(
        const RunSection::General::Resonance::
            HybridNuclearResonancePointProvider &provider,
        const RunSection::General::Resonance::
            HybridNuclearResonanceFieldResponseRequest &response,
        const RunSection::General::Resonance::
            SpectrumRequest &request,
        RunSection::General::Resonance::
            ResonanceLineSet &lines,
        std::string &error)
    {
        using namespace RunSection::General::Resonance;
        HybridNuclearResonanceReport report;
        return HybridNuclearResonanceSolver::
            GenerateFirstOrderFiniteDifference(
                provider,response,request,
                lines,report,error);
    }

    bool GRC_GenerateFirstOrderFiniteDifference(
        const RunSection::General::Resonance::
            HybridNuclearResonancePointProvider &provider,
        const RunSection::General::Resonance::
            HybridNuclearResonanceFieldResponseRequest &response,
        const RunSection::General::Resonance::
            SpectrumRequest &request,
        RunSection::General::Resonance::
            ResonanceLineSet &lines,
        RunSection::General::Resonance::
            HybridNuclearResonanceReport &report,
        std::string &error)
    {
        return RunSection::General::Resonance::
            HybridNuclearResonanceSolver::
                GenerateFirstOrderFiniteDifference(
                    provider,response,request,
                    lines,report,error);
    }

    bool GRC_Density(const SpinAPI::system_ptr &system, SpinAPI::SpinSpace &space, arma::cx_mat &rho)
    {
        auto state = system->states_find("Up");
        if (state == nullptr || !space.GetState(state, rho)) return false;
        const arma::cx_double tr = arma::trace(rho);
        if (std::abs(tr) < 1.0e-15) return false;
        rho /= tr;
        return rho.is_finite();
    }

    bool GRC_TransverseOperators(
        const SpinAPI::system_ptr &system,
        SpinAPI::SpinSpace &space,
        const RunSection::General::HS::HSOrientation &orientation,
        arma::cx_mat &muX,arma::cx_mat &muY)
    {
        using namespace RunSection::General::Resonance;

        auto spin = system->spins_find("E");
        auto zeeman = system->interactions_find("B0");
        if (spin == nullptr || zeeman == nullptr)
            return false;

        std::string error;
        return ResonanceMagneticMomentBuilder::BuildTransverse(
            space,
            {{spin,zeeman}},
            orientation.frameToLab,
            true,
            muX,muY,error);
    }

    bool GRC_TransverseOperators(
        const SpinAPI::system_ptr &system,
        SpinAPI::SpinSpace &space,
        arma::cx_mat &muX,arma::cx_mat &muY)
    {
        return GRC_TransverseOperators(
            system,space,GRC_IdentityOrientation(),
            muX,muY);
    }

    bool GRC_ParseLegacyColumns(const std::string &data, std::vector<double> &fields,
                                std::vector<double> &signal)
    {
        std::istringstream in(data);
        std::string line;
        if (!std::getline(in,line)) return false;
        std::vector<std::string> headers;
        {
            std::istringstream hs(line);
            for (std::string token; hs >> token;) headers.push_back(token);
        }
        auto fit = std::find(headers.begin(), headers.end(), "System.Field_mT");
        auto sit = std::find(headers.begin(), headers.end(), "System.Total_perp");
        if (fit == headers.end() || sit == headers.end()) return false;
        const size_t fi = static_cast<size_t>(std::distance(headers.begin(), fit));
        const size_t si = static_cast<size_t>(std::distance(headers.begin(), sit));

        fields.clear(); signal.clear();
        while (std::getline(in,line))
        {
            if (line.empty()) continue;
            std::istringstream ls(line);
            std::vector<double> row;
            for (std::string token; ls >> token;)
            {
                try { row.push_back(std::stod(token)); }
                catch (...) { return false; }
            }
            if (row.size() <= std::max(fi,si)) return false;
            fields.push_back(row[fi]);
            signal.push_back(row[si]);
        }
        return !fields.empty() && fields.size() == signal.size();
    }

    bool GRC_LegacySweep(const GRC_Model &model, double startT, double stepT, int steps,
                         double frequencyGHz, double linewidth_mT,
                         std::vector<double> &fields_mT, std::vector<double> &signal)
    {
        auto system = GRC_BuildSystem(model, startT);
        if (system == nullptr) return false;

        RunSection::RunSection rs;
        rs.Add(system);
        MSDParser::ObjectParser settings("general", "steps="+std::to_string(steps)+";");
        rs.Add(MSDParser::ObjectType::Settings, settings);

        std::ostringstream props;
        props << std::setprecision(17)
              << "type=statichs-resonance-spectra;mwfrequency=" << frequencyGHz
              << ";linewidth=" << linewidth_mT
              << ";lineshape=gaussian;detectspins=E;fieldinteraction=B0;hamiltonianh0list=B0";
        if (model.hyperfine != 0.0) props << ",A";
        props << ";powdersamplingpoints=1;powdergridtype=uniform;powdergammapoints=1;"
                 "powderfullsphere=true;fulltensorrotation=true;sweepcache=false;initialstate=Up;";
        MSDParser::ObjectParser taskParser("res", props.str());
        if (!rs.Add(MSDParser::ObjectType::Task, taskParser)) return false;
        auto task = rs.GetTask("res");
        if (task == nullptr) return false;

        std::ostringstream actionProps;
        actionProps << std::setprecision(17)
                    << "type=addvector;vector=System.B0.field;direction=0 0 1;value=" << stepT << ";";
        MSDParser::ObjectParser action("field", actionProps.str());
        if (!rs.Add(MSDParser::ObjectType::Action, action)) return false;

        std::ostringstream log, data;
        data << std::setprecision(17);
        task->SetLogStream(log);
        task->SetDataStream(data);
        for (int s=1; s<=steps; ++s)
        {
            if (!rs.Run(static_cast<unsigned int>(s))) return false;
            if (s < steps && !rs.Step(static_cast<unsigned int>(s+1))) return false;
        }
        return GRC_ParseLegacyColumns(data.str(), fields_mT, signal);
    }

    bool GRC_GeneralSweep(const GRC_Model &model, double startT, double stepT, int steps,
                          double frequencyGHz, double linewidth_mT,
                          std::vector<double> &fields_mT, std::vector<double> &signal)
    {
        fields_mT.clear(); signal.clear();
        const auto orientation = GRC_IdentityOrientation();
        for (int i=0; i<steps; ++i)
        {
            const double fieldT = startT + static_cast<double>(i)*stepT;
            auto system = GRC_BuildSystem(model, fieldT);
            if (system == nullptr) return false;
            SpinAPI::SpinSpace space(*system);
            space.UseSuperoperatorSpace(false);
            space.UseFullTensorRotation(true);

            const auto plan = GRC_Plan(model, false);
            RunSection::General::Resonance::GeneralResonanceHamiltonian hbuilder(plan, space);
            arma::sp_cx_mat H, dHdB;
            std::string error;
            if (!hbuilder.Build(orientation, H, error)) return false;
            if (!hbuilder.BuildFieldDerivative(orientation, {"B0"}, fieldT, dHdB, error)) return false;

            arma::cx_mat rho, muX, muY;
            if (!GRC_Density(system, space, rho) || !GRC_TransverseOperators(
                system,space,orientation, muX, muY))
                return false;

            RunSection::General::Resonance::SpectrumRequest request;
            request.microwaveFrequencyGHz = frequencyGHz;
            request.linewidth_mT = linewidth_mT;
            request.lineshape = RunSection::General::Resonance::Lineshape::Gaussian;
            RunSection::General::Resonance::SpectrumPoint point;
            if (!RunSection::General::Resonance::ResonanceSpectrumEvaluator::Evaluate(
                    H, rho, dHdB, muX, muY, request, point, error))
                return false;
            fields_mT.push_back(1.0e3*fieldT);
            signal.push_back(point.totalPerpendicular);
        }
        return true;
    }

    double GRC_AtLegacyOutputPrecision(double value)
    {
        // Frozen StaticHS-Resonance-Spectra writes through the task data
        // stream at the historical default precision (six significant digits).
        // Quantize the General result through the same representation before
        // comparing the public task output; this avoids mistaking stream
        // formatting for a physics discrepancy.
        std::ostringstream out;
        out << std::setprecision(6) << value;
        return std::stod(out.str());
    }

    bool GRC_CompareLegacyAndGeneral(const GRC_Model &model, double centerT,
                                     double spanT, double stepT,
                                     double frequencyGHz, double linewidth_mT)
    {
        const int steps = static_cast<int>(std::llround((2.0*spanT)/stepT))+1;
        const double startT = centerT-spanT;
        std::vector<double> lf, ls, gf, gs;
        if (!GRC_LegacySweep(model,startT,stepT,steps,frequencyGHz,linewidth_mT,lf,ls) ||
            !GRC_GeneralSweep(model,startT,stepT,steps,frequencyGHz,linewidth_mT,gf,gs))
            return false;
        if (lf.size()!=gf.size() || ls.size()!=gs.size()) return false;

        for (size_t i=0;i<ls.size();++i)
        {
            const double gfLegacy = GRC_AtLegacyOutputPrecision(gf[i]);
            const double gsLegacy = GRC_AtLegacyOutputPrecision(gs[i]);
            if (std::abs(lf[i]-gfLegacy) > 1.0e-12) return false;
            if (std::abs(ls[i]-gsLegacy) > 1.0e-12*std::max(1.0,std::abs(ls[i]))) return false;
        }
        return true;
    }

    bool GRC_TestLineshapeContract()
    {
        using namespace RunSection::General::Resonance;
        const double width=0.4;
        const double g0=ResonanceLineshape::Evaluate(Lineshape::Gaussian,0.0,width);
        const double gh=ResonanceLineshape::Evaluate(Lineshape::Gaussian,0.5*width,width);
        const double l0=ResonanceLineshape::Evaluate(Lineshape::Lorentzian,0.0,width);
        const double lh=ResonanceLineshape::Evaluate(Lineshape::Lorentzian,0.5*width,width);
        return g0>0.0 && l0>0.0 && std::abs(gh/g0-0.5)<1.0e-13 &&
               std::abs(lh/l0-0.5)<1.0e-13;
    }

    bool GRC_TestFieldJacobianAndDetector()
    {
        GRC_Model model;
        const double frequency=9.5;
        const double omega=2.0*arma::datum::pi*frequency;
        const double fieldT=omega/(GRC_MU_B_OVER_HBAR*model.gz);
        auto system=GRC_BuildSystem(model,fieldT);
        if (system==nullptr) return false;
        SpinAPI::SpinSpace space(*system);
        const auto plan=GRC_Plan(model,false);
        const auto orientation=GRC_IdentityOrientation();
        RunSection::General::Resonance::GeneralResonanceHamiltonian builder(plan,space);
        arma::sp_cx_mat H,dHdB;
        std::string error;
        if (!builder.Build(orientation,H,error) ||
            !builder.BuildFieldDerivative(orientation,{"B0"},fieldT,dHdB,error)) return false;
        arma::vec energies;
        arma::cx_mat U;
        if (!arma::eig_sym(energies,U,arma::cx_mat(H))) return false;
        arma::vec slopes;
        if (!RunSection::General::Resonance::ResonanceFieldJacobian::DiagonalEnergyDerivatives(U,dHdB,slopes,error)) return false;
        arma::cx_mat rho;
        if (!GRC_Density(system,space,rho)) return false;
        const arma::cx_mat rhoEigen = U.t()*rho*U;
        arma::vec pops=arma::real(rhoEigen.diag());
        std::vector<RunSection::General::Resonance::Transition> transitions;
        if (!RunSection::General::Resonance::ResonanceTransitionDetector::Detect(
                energies,pops,slopes,omega,transitions,error)) return false;
        if (transitions.size()!=1) return false;
        const auto &t=transitions.front();
        return std::abs(t.detuningOmega)<1.0e-10 &&
               std::abs(t.dOmegaDB-GRC_MU_B_OVER_HBAR*model.gz)<1.0e-10 &&
               std::abs(t.detuningField_mT)<1.0e-9;
    }

    bool GRC_TestDegenerateFieldJacobian()
    {
        using RunSection::General::Resonance::ResonanceFieldJacobian;
        arma::vec energies(2,arma::fill::zeros);
        arma::cx_mat eigenvectors=arma::eye<arma::cx_mat>(2,2);
        arma::sp_cx_mat derivative(2,2);
        derivative(0,1)=2.5;
        derivative(1,0)=2.5;
        arma::vec slopes;
        std::string error;
        if(!ResonanceFieldJacobian::ResolveDegenerateSubspaces(
                energies,eigenvectors,derivative,slopes,error)) return false;
        const arma::cx_mat resolved=eigenvectors.t()*arma::cx_mat(derivative)*eigenvectors;
        return slopes.n_elem==2 && std::abs(slopes(0)+2.5)<1.0e-12 &&
               std::abs(slopes(1)-2.5)<1.0e-12 &&
               arma::norm(resolved-arma::diagmat(arma::conv_to<arma::cx_vec>::from(slopes)),"fro")<1.0e-12;
    }

    bool GRC_TestHamiltonianAdapterPreservesApproximation()
    {
        GRC_Model model{2.0,2.1,2.4,0.0};
        auto system=GRC_BuildSystem(model,0.34);
        if (system==nullptr) return false;
        SpinAPI::SpinSpace space(*system);
        RunSection::General::HS::HSOrientation orientation;
        orientation.alpha=0.31; orientation.beta=0.67; orientation.gamma=-0.28;
        SpinAPI::CreateZYZRotationMatrix(orientation.alpha,orientation.beta,orientation.gamma,orientation.frameToLab);

        const auto fullPlan=GRC_Plan(model,false);
        const auto secPlan=GRC_Plan(model,true);
        RunSection::General::Resonance::GeneralResonanceHamiltonian fullBuilder(fullPlan,space);
        RunSection::General::Resonance::GeneralResonanceHamiltonian secBuilder(secPlan,space);
        arma::sp_cx_mat Hfull,Hsec,dFull,dSec;
        std::string error;
        if (!fullBuilder.Build(orientation,Hfull,error) || !secBuilder.Build(orientation,Hsec,error) ||
            !fullBuilder.BuildFieldDerivative(orientation,{"B0"},0.34,dFull,error) ||
            !secBuilder.BuildFieldDerivative(orientation,{"B0"},0.34,dSec,error)) return false;
        if (arma::norm(arma::cx_mat(Hfull-Hsec),"fro")<1.0e-5) return false;
        const arma::cx_mat HfullDense(Hfull), HsecDense(Hsec), dFullDense(dFull), dSecDense(dSec);
        if (std::abs(HsecDense(0,1))>1.0e-11 || std::abs(dSecDense(0,1))>1.0e-11) return false;
        return std::abs(HfullDense(0,1))>1.0e-6 && std::abs(dFullDense(0,1))>1.0e-6;
    }

    struct GRC_I72Comparison
    {
        RunSection::General::Resonance::ResonanceLineSet core;
        RunSection::General::Resonance::ResonanceLineSet exact;
        RunSection::General::Resonance::ResonanceLineSet hybrid;
    };

    SpinAPI::system_ptr GRC_BuildI72System(double fieldT, double hyperfine)
    {
        std::ostringstream eprops;
        eprops << std::setprecision(17)
               << "type=electron;spin=1/2;tensor=isotropic(2.0023);";
        auto electron = std::make_shared<SpinAPI::Spin>("E",eprops.str());
        auto nucleus = std::make_shared<SpinAPI::Spin>(
            "V","type=nucleus;spin=7/2;tensor=isotropic(1);");

        std::ostringstream bprops;
        bprops << std::setprecision(17)
               << "type=zeeman;spins=E;field=0 0 " << fieldT
               << ";ignoretensors=false;commonprefactor=true;prefactor=1.0;";
        auto field = std::make_shared<SpinAPI::Interaction>("B0",bprops.str());

        std::ostringstream aprops;
        aprops << std::setprecision(17)
               << "type=hyperfine;group1=E;group2=V;tensor=isotropic("
               << hyperfine
               << ");ignoretensors=true;commonprefactor=false;prefactor=1.0;";
        auto hfc = std::make_shared<SpinAPI::Interaction>("A",aprops.str());

        // ^51V gamma/(2*pi) is about 11.2 MHz/T. The synthetic benchmark uses
        // 0.0703 rad/ns/T explicitly because the current generic SpinAPI common
        // prefactor is electronic; nuclear Zeeman must not reuse mu_B/hbar.
        std::ostringstream nzprops;
        nzprops << std::setprecision(17)
                << "type=zeeman;spins=V;field=0 0 " << fieldT
                << ";ignoretensors=true;commonprefactor=false;"
                   "prefactor=0.0703;";
        auto nuclearZeeman =
            std::make_shared<SpinAPI::Interaction>("NZ",nzprops.str());

        auto up = std::make_shared<SpinAPI::State>(
            "Up","spin(E)=|1/2>;");

        auto system = std::make_shared<SpinAPI::SpinSystem>("I72");
        system->Add(electron);
        system->Add(nucleus);
        system->Add(field);
        system->Add(hfc);
        system->Add(nuclearZeeman);
        system->Add(up);
        system->SetProperties(
            std::make_shared<MSDParser::ObjectParser>(
                "properties","initialstate=Up;"));

        if (!system->ValidateInteractions().empty())
            return nullptr;
        if (!up->ParseFromSystem(*system))
            return nullptr;
        return system;
    }

    RunSection::General::HS::HSExecutionPlan GRC_H0Plan(
        const std::vector<std::string> &names)
    {
        RunSection::General::HS::HSExecutionPlan plan;
        plan.dynamics = RunSection::General::HS::Dynamics::Static;
        plan.calculation =
            RunSection::General::HS::Calculation::TimeEvolution;
        plan.sampling = RunSection::General::HS::Sampling::Direct;
        plan.orientation =
            RunSection::General::HS::OrientationMode::Explicit;
        plan.approximation = SpinAPI::HamiltonianApproximation::Full;
        plan.hasH0List = true;
        plan.h0List = names;
        return plan;
    }

    bool GRC_BuildI72Comparison(double frequencyGHz, double hyperfine,
        bool qualifyProjection, GRC_I72Comparison &result)
    {
        using namespace RunSection::General::Resonance;

        const double omega = 2.0*arma::datum::pi*frequencyGHz;
        const double fieldT =
            omega/(GRC_MU_B_OVER_HBAR*2.0023);
        auto system = GRC_BuildI72System(fieldT,hyperfine);
        if (system == nullptr)
            return false;

        auto electron = system->spins_find("E");
        auto nucleus = system->spins_find("V");
        auto b0 = system->interactions_find("B0");
        auto hfc = system->interactions_find("A");
        auto nz = system->interactions_find("NZ");
        if (electron == nullptr || nucleus == nullptr ||
            b0 == nullptr || hfc == nullptr || nz == nullptr)
            return false;

        const auto orientation = GRC_IdentityOrientation();
        arma::mat rotation = orientation.frameToLab;
        std::string error;

        SpectrumRequest request;
        request.microwaveFrequencyGHz = frequencyGHz;
        request.linewidth_mT = 0.10;
        request.lineshape = Lineshape::Gaussian;
        request.populationThreshold = 1.0e-15;
        request.minimumSlope = 1.0e-15;

        // Complete exact electron+nucleus reference.
        SpinAPI::SpinSpace fullSpace(*system);
        fullSpace.UseSuperoperatorSpace(false);
        fullSpace.UseFullTensorRotation(true);
        const auto fullPlan = GRC_H0Plan({"B0","A","NZ"});
        GeneralResonanceHamiltonian fullBuilder(fullPlan,fullSpace);

        arma::sp_cx_mat fullH,fullDHdB;
        arma::cx_mat fullRho,fullMuX,fullMuY;
        if (!fullBuilder.Build(orientation,fullH,error) ||
            !fullBuilder.BuildFieldDerivative(
                orientation,{"B0","NZ"},fieldT,fullDHdB,error) ||
            !GRC_Density(system,fullSpace,fullRho) ||
            !GRC_TransverseOperators(
                system,fullSpace,fullMuX,fullMuY) ||
            !ExactResonanceSolver::Generate(
                fullH,fullRho,fullDHdB,fullMuX,fullMuY,
                request,result.exact,error))
            return false;

        // Exact electronic/core problem. The perturbative nucleus is absent.
        SpinAPI::SpinSpace coreSpace(
            std::vector<SpinAPI::spin_ptr>{electron});
        coreSpace.Add(b0);
        coreSpace.UseSuperoperatorSpace(false);
        coreSpace.UseFullTensorRotation(true);
        const auto corePlan = GRC_H0Plan({"B0"});
        GeneralResonanceHamiltonian coreBuilder(corePlan,coreSpace);

        arma::sp_cx_mat coreH,coreDHdB;
        arma::cx_mat coreRho,coreMuX,coreMuY;
        if (!coreBuilder.Build(orientation,coreH,error) ||
            !coreBuilder.BuildFieldDerivative(
                orientation,{"B0"},fieldT,coreDHdB,error) ||
            !GRC_Density(system,coreSpace,coreRho) ||
            !GRC_TransverseOperators(
                system,coreSpace,coreMuX,coreMuY) ||
            !ExactResonanceSolver::Generate(
                coreH,coreRho,coreDHdB,coreMuX,coreMuY,
                request,result.core,error))
            return false;

        // SpinAPI owns hyperfine tensor rotation and prefactors. The
        // perturbative nucleus is deliberately the final Kronecker factor.
        SpinAPI::SpinSpace pairSpace(
            std::vector<SpinAPI::spin_ptr>{electron,nucleus});
        pairSpace.UseSuperoperatorSpace(false);
        pairSpace.UseFullTensorRotation(true);
        arma::sp_cx_mat hyperfinePair;
        if (!pairSpace.InteractionOperatorRotatedZYZ(
                hfc,rotation,hyperfinePair))
            return false;

        // One-nucleus explicit Hamiltonian/field derivative.
        SpinAPI::SpinSpace nuclearSpace(nucleus);
        nuclearSpace.UseSuperoperatorSpace(false);
        nuclearSpace.UseFullTensorRotation(true);
        arma::sp_cx_mat nuclearH;
        if (!nuclearSpace.InteractionOperatorRotatedZYZ(
                nz,rotation,nuclearH))
            return false;

        HybridNuclearResonanceNucleus hybrid;
        hybrid.hyperfineCoreNuclear = arma::cx_mat(hyperfinePair);
        hybrid.nuclearHamiltonian = arma::cx_mat(nuclearH);
        hybrid.nuclearDHdB = nuclearH/fieldT;
        hybrid.nuclearDimension =
            static_cast<arma::uword>(nucleus->Multiplicity());
        hybrid.overlapThreshold = 1.0e-14;
        hybrid.fieldIndependentProjection = qualifyProjection;

        return GRC_GenerateFirstOrder(
            coreH,coreRho,coreDHdB,coreMuX,coreMuY,
            hybrid,request,result.hybrid,error);
    }

    double GRC_LineWeight(
        const RunSection::General::Resonance::ResonanceLine &line)
    {
        return std::abs(line.populationDifference)*
               line.moment.perpendicular;
    }

    std::vector<double> GRC_StrongestFrequencies(
        const RunSection::General::Resonance::ResonanceLineSet &set,
        std::size_t count)
    {
        std::vector<std::pair<double,double>> weighted;
        for (const auto &line : set.lines)
        {
            const double weight = GRC_LineWeight(line);
            if (weight > 1.0e-16)
                weighted.push_back({weight,line.omega});
        }
        std::sort(weighted.begin(),weighted.end(),
            [](const auto &a,const auto &b)
            {
                return a.first>b.first;
            });
        if (weighted.size()>count)
            weighted.resize(count);

        std::vector<double> result;
        for (const auto &item : weighted)
            result.push_back(item.second);
        std::sort(result.begin(),result.end());
        return result;
    }

    double GRC_StrongLineFrequencyError(
        const RunSection::General::Resonance::ResonanceLineSet &exact,
        const RunSection::General::Resonance::ResonanceLineSet &hybrid,
        std::size_t count)
    {
        const auto e = GRC_StrongestFrequencies(exact,count);
        const auto h = GRC_StrongestFrequencies(hybrid,count);
        if (e.size()!=count || h.size()!=count)
            return -1.0;

        double error = 0.0;
        for (std::size_t i=0;i<count;++i)
            error = std::max(error,std::abs(e[i]-h[i]));
        return error;
    }

    bool GRC_TestHybridI72ZeroLimit()
    {
        GRC_I72Comparison data;
        if (!GRC_BuildI72Comparison(9.5,0.0,true,data))
            return false;
        if (!data.hybrid.fieldJacobianQualified)
            return false;

        const auto core = GRC_StrongestFrequencies(data.core,1);
        const auto hybrid = GRC_StrongestFrequencies(data.hybrid,8);
        if (core.size()!=1 || hybrid.size()!=8)
            return false;
        for (double omega : hybrid)
            if (std::abs(omega-core.front())>1.0e-12)
                return false;

        double coreWeight=0.0,hybridWeight=0.0;
        for (const auto &line:data.core.lines)
            coreWeight += GRC_LineWeight(line);
        for (const auto &line:data.hybrid.lines)
            hybridWeight += GRC_LineWeight(line);

        return std::abs(coreWeight-hybridWeight) <=
            1.0e-12*std::max(1.0,std::abs(coreWeight));
    }

    bool GRC_TestHybridI72FirstOrderScaling()
    {
        GRC_I72Comparison strong,weak;
        if (!GRC_BuildI72Comparison(9.5,0.40,true,strong) ||
            !GRC_BuildI72Comparison(9.5,0.20,true,weak))
            return false;

        const double eStrong =
            GRC_StrongLineFrequencyError(strong.exact,strong.hybrid,8);
        const double eWeak =
            GRC_StrongLineFrequencyError(weak.exact,weak.hybrid,8);
        if (!(eStrong>0.0) || !(eWeak>0.0))
            return false;

        const double ratio=eStrong/eWeak;
        return ratio>3.5 && ratio<4.5;
    }

    bool GRC_TestHybridI72HighFieldImproves()
    {
        GRC_I72Comparison xband,wband;
        if (!GRC_BuildI72Comparison(9.5,0.40,true,xband) ||
            !GRC_BuildI72Comparison(94.0,0.40,true,wband))
            return false;

        const double eX =
            GRC_StrongLineFrequencyError(xband.exact,xband.hybrid,8);
        const double eW =
            GRC_StrongLineFrequencyError(wband.exact,wband.hybrid,8);
        return eX>0.0 && eW>0.0 && eW<0.2*eX;
    }

    bool GRC_TestHybridUnqualifiedJacobianFailsClosed()
    {
        using namespace RunSection::General::Resonance;
        GRC_I72Comparison data;
        if (!GRC_BuildI72Comparison(9.5,0.40,false,data))
            return false;
        if (data.hybrid.fieldJacobianQualified)
            return false;

        SpectrumRequest request;
        request.microwaveFrequencyGHz=9.5;
        request.linewidth_mT=0.1;
        SpectrumPoint point;
        std::string error;
        if (ResonanceSpectrumEvaluator::Evaluate(
                data.hybrid,request,point,error))
            return false;
        return error ==
            "resonance line set does not have a qualified field Jacobian";
    }

    struct GRC_AnisotropicI72Model
    {
        double gx = 1.94;
        double gy = 2.08;
        double gz = 2.31;
        double ax = 0.22;
        double ay = 0.37;
        double az = 0.61;
        double aAlpha = 0.37;
        double aBeta = 0.58;
        double aGamma = -0.29;
        double bAlpha = 0.0;
        double bBeta = 0.0;
        double bGamma = 0.0;
    };

    SpinAPI::system_ptr GRC_BuildAnisotropicI72System(
        const GRC_AnisotropicI72Model &model,
        double fieldT, double hyperfineScale)
    {
        std::ostringstream eprops;
        eprops << std::setprecision(17)
               << "type=electron;spin=1/2;tensor=anisotropic("
               << model.gx << " " << model.gy << " " << model.gz << ");";
        auto electron = std::make_shared<SpinAPI::Spin>("E",eprops.str());
        auto nucleus = std::make_shared<SpinAPI::Spin>(
            "V","type=nucleus;spin=7/2;tensor=isotropic(1);");

        std::ostringstream bprops;
        bprops << std::setprecision(17)
               << "type=zeeman;spins=E;field=0 0 " << fieldT
               << ";orientation=" << model.bAlpha << ","
               << model.bBeta << "," << model.bGamma
               << ";ignoretensors=false;commonprefactor=true;prefactor=1.0;";
        auto field = std::make_shared<SpinAPI::Interaction>("B0",bprops.str());

        std::ostringstream aprops;
        aprops << std::setprecision(17)
               << "type=hyperfine;group1=E;group2=V;tensor=anisotropic("
               << hyperfineScale*model.ax << " "
               << hyperfineScale*model.ay << " "
               << hyperfineScale*model.az << ");orientation="
               << model.aAlpha << "," << model.aBeta << "," << model.aGamma
               << ";ignoretensors=true;commonprefactor=false;prefactor=1.0;";
        auto hfc = std::make_shared<SpinAPI::Interaction>("A",aprops.str());

        std::ostringstream nzprops;
        nzprops << std::setprecision(17)
                << "type=zeeman;spins=V;field=0 0 " << fieldT
                << ";ignoretensors=true;commonprefactor=false;"
                   "prefactor=0.0703;";
        auto nuclearZeeman =
            std::make_shared<SpinAPI::Interaction>("NZ",nzprops.str());

        auto up = std::make_shared<SpinAPI::State>(
            "Up","spin(E)=|1/2>;");

        auto system = std::make_shared<SpinAPI::SpinSystem>("I72Aniso");
        system->Add(electron);
        system->Add(nucleus);
        system->Add(field);
        system->Add(hfc);
        system->Add(nuclearZeeman);
        system->Add(up);
        system->SetProperties(
            std::make_shared<MSDParser::ObjectParser>(
                "properties","initialstate=Up;"));

        if (!system->ValidateInteractions().empty())
            return nullptr;
        if (!up->ParseFromSystem(*system))
            return nullptr;
        return system;
    }

    bool GRC_BuildAnisotropicI72Comparison(
        double frequencyGHz, double hyperfineScale,
        const GRC_AnisotropicI72Model &model,
        const RunSection::General::HS::HSOrientation &orientation,
        GRC_I72Comparison &result)
    {
        using namespace RunSection::General::Resonance;

        const double omega = 2.0*arma::datum::pi*frequencyGHz;
        const double fieldT =
            omega/(GRC_MU_B_OVER_HBAR*model.gz);

        auto system =
            GRC_BuildAnisotropicI72System(model,fieldT,hyperfineScale);
        if (system == nullptr)
            return false;

        auto electron = system->spins_find("E");
        auto nucleus = system->spins_find("V");
        auto b0 = system->interactions_find("B0");
        auto hfc = system->interactions_find("A");
        auto nz = system->interactions_find("NZ");
        if (electron == nullptr || nucleus == nullptr ||
            b0 == nullptr || hfc == nullptr || nz == nullptr)
            return false;

        arma::mat rotation = orientation.frameToLab;
        std::string error;

        SpectrumRequest request;
        request.microwaveFrequencyGHz = frequencyGHz;
        request.linewidth_mT = 0.10;
        request.lineshape = Lineshape::Gaussian;
        request.populationThreshold = 1.0e-15;
        request.minimumSlope = 1.0e-15;

        SpinAPI::SpinSpace fullSpace(*system);
        fullSpace.UseSuperoperatorSpace(false);
        fullSpace.UseFullTensorRotation(true);
        const auto fullPlan = GRC_H0Plan({"B0","A","NZ"});
        GeneralResonanceHamiltonian fullBuilder(fullPlan,fullSpace);

        arma::sp_cx_mat fullH,fullDHdB;
        arma::cx_mat fullRho,fullMuX,fullMuY;
        if (!fullBuilder.Build(orientation,fullH,error) ||
            !fullBuilder.BuildFieldDerivative(
                orientation,{"B0","NZ"},fieldT,fullDHdB,error) ||
            !GRC_Density(system,fullSpace,fullRho) ||
            !GRC_TransverseOperators(
                system,fullSpace,orientation,fullMuX,fullMuY) ||
            !ExactResonanceSolver::Generate(
                fullH,fullRho,fullDHdB,fullMuX,fullMuY,
                request,result.exact,error))
            return false;

        SpinAPI::SpinSpace coreSpace(
            std::vector<SpinAPI::spin_ptr>{electron});
        coreSpace.Add(b0);
        coreSpace.UseSuperoperatorSpace(false);
        coreSpace.UseFullTensorRotation(true);
        const auto corePlan = GRC_H0Plan({"B0"});
        GeneralResonanceHamiltonian coreBuilder(corePlan,coreSpace);

        arma::sp_cx_mat coreH,coreDHdB;
        arma::cx_mat coreRho,coreMuX,coreMuY;
        if (!coreBuilder.Build(orientation,coreH,error) ||
            !coreBuilder.BuildFieldDerivative(
                orientation,{"B0"},fieldT,coreDHdB,error) ||
            !GRC_Density(system,coreSpace,coreRho) ||
            !GRC_TransverseOperators(
                system,coreSpace,orientation,coreMuX,coreMuY) ||
            !ExactResonanceSolver::Generate(
                coreH,coreRho,coreDHdB,coreMuX,coreMuY,
                request,result.core,error))
            return false;

        SpinAPI::SpinSpace pairSpace(
            std::vector<SpinAPI::spin_ptr>{electron,nucleus});
        pairSpace.UseSuperoperatorSpace(false);
        pairSpace.UseFullTensorRotation(true);
        arma::sp_cx_mat hyperfinePair;
        if (!pairSpace.InteractionOperatorRotatedZYZ(
                hfc,rotation,hyperfinePair))
            return false;

        SpinAPI::SpinSpace nuclearSpace(nucleus);
        nuclearSpace.UseSuperoperatorSpace(false);
        nuclearSpace.UseFullTensorRotation(true);
        arma::sp_cx_mat nuclearH;
        if (!nuclearSpace.InteractionOperatorRotatedZYZ(
                nz,rotation,nuclearH))
            return false;

        HybridNuclearResonanceNucleus hybrid;
        hybrid.hyperfineCoreNuclear = arma::cx_mat(hyperfinePair);
        hybrid.nuclearHamiltonian = arma::cx_mat(nuclearH);
        hybrid.nuclearDHdB = nuclearH/fieldT;
        hybrid.nuclearDimension =
            static_cast<arma::uword>(nucleus->Multiplicity());
        hybrid.overlapThreshold = 1.0e-14;
        hybrid.fieldIndependentProjection = true;

        return GRC_GenerateFirstOrder(
            coreH,coreRho,coreDHdB,coreMuX,coreMuY,
            hybrid,request,result.hybrid,error);
    }

    RunSection::General::HS::HSOrientation GRC_Orientation(
        double alpha,double beta,double gamma)
    {
        RunSection::General::HS::HSOrientation orientation;
        orientation.alpha=alpha;
        orientation.beta=beta;
        orientation.gamma=gamma;
        orientation.weight=1.0;
        SpinAPI::CreateZYZRotationMatrix(
            alpha,beta,gamma,orientation.frameToLab);
        return orientation;
    }

    bool GRC_TestHybridAnisotropicA2ScalingAtArbitraryOrientation()
    {
        GRC_AnisotropicI72Model model;
        const auto orientation=GRC_Orientation(0.43,0.71,-0.33);

        GRC_I72Comparison strong,weak;
        if (!GRC_BuildAnisotropicI72Comparison(
                9.5,1.0,model,orientation,strong) ||
            !GRC_BuildAnisotropicI72Comparison(
                9.5,0.5,model,orientation,weak))
            return false;

        const double eStrong =
            GRC_StrongLineFrequencyError(strong.exact,strong.hybrid,8);
        const double eWeak =
            GRC_StrongLineFrequencyError(weak.exact,weak.hybrid,8);
        if (!(eStrong>0.0) || !(eWeak>0.0))
            return false;

        const double ratio=eStrong/eWeak;
        if (!(ratio>3.2 && ratio<4.8))
            return false;

        GRC_I72Comparison identity;
        if (!GRC_BuildAnisotropicI72Comparison(
                9.5,1.0,model,GRC_IdentityOrientation(),identity))
            return false;
        const auto a=GRC_StrongestFrequencies(strong.hybrid,8);
        const auto b=GRC_StrongestFrequencies(identity.hybrid,8);
        if (a.size()!=8 || b.size()!=8)
            return false;

        double orientationShift=0.0;
        for (std::size_t i=0;i<8;++i)
            orientationShift=std::max(
                orientationShift,std::abs(a[i]-b[i]));
        return orientationShift>1.0e-4;
    }

    bool GRC_TestHybridAnisotropicXWImprovement()
    {
        GRC_AnisotropicI72Model model;
        const auto orientation=GRC_Orientation(-0.27,0.83,0.52);

        GRC_I72Comparison xband,wband;
        if (!GRC_BuildAnisotropicI72Comparison(
                9.5,1.0,model,orientation,xband) ||
            !GRC_BuildAnisotropicI72Comparison(
                94.0,1.0,model,orientation,wband))
            return false;

        const double eX =
            GRC_StrongLineFrequencyError(xband.exact,xband.hybrid,8);
        const double eW =
            GRC_StrongLineFrequencyError(wband.exact,wband.hybrid,8);
        return eX>0.0 && eW>0.0 && eW<0.25*eX;
    }

    bool GRC_TestHybridNonCoaxialTensorFrameIsActive()
    {
        GRC_AnisotropicI72Model noncoaxial;
        GRC_AnisotropicI72Model coaxial=noncoaxial;
        coaxial.aAlpha=0.0;
        coaxial.aBeta=0.0;
        coaxial.aGamma=0.0;

        const auto orientation=GRC_Orientation(0.31,0.64,-0.41);
        GRC_I72Comparison a,b;
        if (!GRC_BuildAnisotropicI72Comparison(
                9.5,1.0,noncoaxial,orientation,a) ||
            !GRC_BuildAnisotropicI72Comparison(
                9.5,1.0,coaxial,orientation,b))
            return false;

        const auto fa=GRC_StrongestFrequencies(a.hybrid,8);
        const auto fb=GRC_StrongestFrequencies(b.hybrid,8);
        if (fa.size()!=8 || fb.size()!=8)
            return false;

        double shift=0.0;
        for (std::size_t i=0;i<8;++i)
            shift=std::max(shift,std::abs(fa[i]-fb[i]));
        return shift>1.0e-4;
    }

    bool GRC_TestHybridRigidZFramePowderComposition()
    {
        const double phi=0.39;
        const double aRelative=0.24;

        GRC_AnisotropicI72Model powderModel;
        powderModel.aAlpha=aRelative;
        powderModel.aBeta=0.0;
        powderModel.aGamma=0.0;

        GRC_AnisotropicI72Model frameModel=powderModel;
        frameModel.bAlpha=-phi;
        frameModel.aAlpha=aRelative-phi;

        GRC_I72Comparison powder,frame;
        if (!GRC_BuildAnisotropicI72Comparison(
                9.5,1.0,powderModel,
                GRC_Orientation(phi,0.0,0.0),powder) ||
            !GRC_BuildAnisotropicI72Comparison(
                9.5,1.0,frameModel,
                GRC_IdentityOrientation(),frame))
            return false;

        const auto ep=GRC_StrongestFrequencies(powder.exact,8);
        const auto ef=GRC_StrongestFrequencies(frame.exact,8);
        const auto hp=GRC_StrongestFrequencies(powder.hybrid,8);
        const auto hf=GRC_StrongestFrequencies(frame.hybrid,8);
        if (ep.size()!=8 || ef.size()!=8 ||
            hp.size()!=8 || hf.size()!=8)
            return false;

        double exactError=0.0,hybridError=0.0;
        for (std::size_t i=0;i<8;++i)
        {
            exactError=std::max(exactError,std::abs(ep[i]-ef[i]));
            hybridError=std::max(hybridError,std::abs(hp[i]-hf[i]));
        }
        return exactError<1.0e-11 && hybridError<1.0e-11;
    }

    struct GRC_ZfsHybridModel
    {
        std::string electronSpin = "1";
        std::string stateKet = "1";
        double gx = 1.91;
        double gy = 2.07;
        double gz = 2.23;
        double d = 22.0;
        double e = 6.0;
        double dAlpha = 0.21;
        double dBeta = 0.49;
        double dGamma = -0.17;
        double ax = 0.18;
        double ay = 0.31;
        double az = 0.52;
        double aAlpha = -0.28;
        double aBeta = 0.63;
        double aGamma = 0.34;
    };

    SpinAPI::system_ptr GRC_BuildZfsHybridSystem(
        const GRC_ZfsHybridModel &model,
        double fieldT,double hyperfineScale)
    {
        std::ostringstream eprops;
        eprops << std::setprecision(17)
               << "type=electron;spin=" << model.electronSpin
               << ";tensor=anisotropic("
               << model.gx << " " << model.gy << " "
               << model.gz << ");";
        auto electron =
            std::make_shared<SpinAPI::Spin>("E",eprops.str());
        auto nucleus = std::make_shared<SpinAPI::Spin>(
            "V","type=nucleus;spin=7/2;tensor=isotropic(1);");

        std::ostringstream bprops;
        bprops << std::setprecision(17)
               << "type=zeeman;spins=E;field=0 0 " << fieldT
               << ";ignoretensors=false;commonprefactor=true;"
                  "prefactor=1.0;";
        auto field =
            std::make_shared<SpinAPI::Interaction>("B0",bprops.str());

        std::ostringstream dprops;
        dprops << std::setprecision(17)
               << "type=zfs;spins=E;dvalue=" << model.d
               << ";evalue=" << model.e
               << ";orientation=" << model.dAlpha << ","
               << model.dBeta << "," << model.dGamma
               << ";energyshift=true;commonprefactor=false;"
                  "prefactor=1.0;";
        auto zfs =
            std::make_shared<SpinAPI::Interaction>("ZFS",dprops.str());

        std::ostringstream aprops;
        aprops << std::setprecision(17)
               << "type=hyperfine;group1=E;group2=V;"
                  "tensor=anisotropic("
               << hyperfineScale*model.ax << " "
               << hyperfineScale*model.ay << " "
               << hyperfineScale*model.az << ");orientation="
               << model.aAlpha << "," << model.aBeta << ","
               << model.aGamma
               << ";ignoretensors=true;commonprefactor=false;"
                  "prefactor=1.0;";
        auto hfc =
            std::make_shared<SpinAPI::Interaction>("A",aprops.str());

        std::ostringstream nzprops;
        nzprops << std::setprecision(17)
                << "type=zeeman;spins=V;field=0 0 " << fieldT
                << ";ignoretensors=true;commonprefactor=false;"
                   "prefactor=0.0703;";
        auto nuclearZeeman =
            std::make_shared<SpinAPI::Interaction>("NZ",nzprops.str());

        std::ostringstream stateProps;
        stateProps << "spin(E)=|" << model.stateKet << ">;";
        auto up = std::make_shared<SpinAPI::State>(
            "Up",stateProps.str());

        auto system =
            std::make_shared<SpinAPI::SpinSystem>("ZfsI72");
        system->Add(electron);
        system->Add(nucleus);
        system->Add(field);
        system->Add(zfs);
        system->Add(hfc);
        system->Add(nuclearZeeman);
        system->Add(up);
        system->SetProperties(
            std::make_shared<MSDParser::ObjectParser>(
                "properties","initialstate=Up;"));

        if (!system->ValidateInteractions().empty())
            return nullptr;
        if (!up->ParseFromSystem(*system))
            return nullptr;
        return system;
    }

    bool GRC_BuildZfsHybridPoint(
        const GRC_ZfsHybridModel &model,
        const RunSection::General::HS::HSOrientation &orientation,
        double hyperfineScale,double fieldT,
        RunSection::General::Resonance::HybridNuclearResonancePoint &point,
        std::string &error)
    {
        using namespace RunSection::General::Resonance;

        auto system =
            GRC_BuildZfsHybridSystem(
                model,fieldT,hyperfineScale);
        if (system == nullptr)
        {
            error = "failed to build synthetic ZFS hybrid system";
            return false;
        }

        auto electron = system->spins_find("E");
        auto nucleus = system->spins_find("V");
        auto b0 = system->interactions_find("B0");
        auto zfs = system->interactions_find("ZFS");
        auto hfc = system->interactions_find("A");
        auto nz = system->interactions_find("NZ");
        if (electron == nullptr || nucleus == nullptr ||
            b0 == nullptr || zfs == nullptr ||
            hfc == nullptr || nz == nullptr)
        {
            error = "missing synthetic ZFS hybrid object";
            return false;
        }

        SpinAPI::SpinSpace coreSpace(
            std::vector<SpinAPI::spin_ptr>{electron});
        coreSpace.Add(b0);
        coreSpace.Add(zfs);
        coreSpace.UseSuperoperatorSpace(false);
        coreSpace.UseFullTensorRotation(true);

        const auto corePlan =
            GRC_H0Plan({"B0","ZFS"});
        GeneralResonanceHamiltonian coreBuilder(
            corePlan,coreSpace);

        if (!coreBuilder.Build(
                orientation,point.coreHamiltonian,error) ||
            !coreBuilder.BuildFieldDerivative(
                orientation,{"B0"},fieldT,
                point.coreDHdB,error) ||
            !GRC_Density(
                system,coreSpace,point.coreDensity) ||
            !GRC_TransverseOperators(
                system,coreSpace,orientation,
                point.coreMuX,point.coreMuY))
            return false;

        arma::mat rotation = orientation.frameToLab;

        SpinAPI::SpinSpace pairSpace(
            std::vector<SpinAPI::spin_ptr>{
                electron,nucleus});
        pairSpace.UseSuperoperatorSpace(false);
        pairSpace.UseFullTensorRotation(true);
        arma::sp_cx_mat hyperfinePair;
        if (!pairSpace.InteractionOperatorRotatedZYZ(
                hfc,rotation,hyperfinePair))
        {
            error = "failed to build synthetic hyperfine operator";
            return false;
        }

        SpinAPI::SpinSpace nuclearSpace(nucleus);
        nuclearSpace.UseSuperoperatorSpace(false);
        nuclearSpace.UseFullTensorRotation(true);
        arma::sp_cx_mat nuclearH;
        if (!nuclearSpace.InteractionOperatorRotatedZYZ(
                nz,rotation,nuclearH))
        {
            error = "failed to build synthetic nuclear Zeeman operator";
            return false;
        }

        point.hybrid.nuclei.resize(1);
        point.hybrid.nuclei.front().hyperfineCoreNuclear =
            arma::cx_mat(hyperfinePair);
        point.hybrid.nuclei.front().nuclearHamiltonian =
            arma::cx_mat(nuclearH);
        point.hybrid.nuclei.front().nuclearDHdB =
            nuclearH/fieldT;
        point.hybrid.nuclei.front().nuclearDimension =
            static_cast<arma::uword>(
                nucleus->Multiplicity());
        point.hybrid.nuclei.front().overlapThreshold=1.0e-14;
        point.hybrid.nuclei.front().fieldIndependentProjection=false;
        return true;
    }

    bool GRC_BuildZfsExactLines(
        const GRC_ZfsHybridModel &model,
        const RunSection::General::HS::HSOrientation &orientation,
        double hyperfineScale,double fieldT,
        double frequencyGHz,
        RunSection::General::Resonance::ResonanceLineSet &lines)
    {
        using namespace RunSection::General::Resonance;

        auto system =
            GRC_BuildZfsHybridSystem(
                model,fieldT,hyperfineScale);
        if (system == nullptr)
            return false;

        SpinAPI::SpinSpace fullSpace(*system);
        fullSpace.UseSuperoperatorSpace(false);
        fullSpace.UseFullTensorRotation(true);
        const auto fullPlan =
            GRC_H0Plan({"B0","ZFS","A","NZ"});
        GeneralResonanceHamiltonian fullBuilder(
            fullPlan,fullSpace);

        arma::sp_cx_mat H,dHdB;
        arma::cx_mat rho,muX,muY;
        std::string error;
        if (!fullBuilder.Build(orientation,H,error) ||
            !fullBuilder.BuildFieldDerivative(
                orientation,{"B0","NZ"},fieldT,dHdB,error) ||
            !GRC_Density(system,fullSpace,rho) ||
            !GRC_TransverseOperators(
                system,fullSpace,orientation,muX,muY))
            return false;

        SpectrumRequest request;
        request.microwaveFrequencyGHz=frequencyGHz;
        request.linewidth_mT=0.10;
        request.populationThreshold=1.0e-15;
        request.minimumSlope=1.0e-15;

        return ExactResonanceSolver::Generate(
            H,rho,dHdB,muX,muY,request,lines,error);
    }

    struct GRC_WeightedSlopeLine
    {
        double weight=0.0;
        double omega=0.0;
        double slope=0.0;
    };

    std::vector<GRC_WeightedSlopeLine>
    GRC_StrongestSlopeLines(
        const RunSection::General::Resonance::ResonanceLineSet &set,
        std::size_t count)
    {
        std::vector<GRC_WeightedSlopeLine> weighted;
        for (const auto &line:set.lines)
        {
            const double weight=GRC_LineWeight(line);
            if (weight>1.0e-16)
                weighted.push_back(
                    {weight,line.omega,line.dOmegaDB});
        }
        std::sort(weighted.begin(),weighted.end(),
            [](const auto &a,const auto &b)
            {
                return a.weight>b.weight;
            });
        if (weighted.size()>count)
            weighted.resize(count);
        std::sort(weighted.begin(),weighted.end(),
            [](const auto &a,const auto &b)
            {
                return a.omega<b.omega;
            });
        return weighted;
    }

    struct GRC_ZfsHybridProductLabel
    {
        arma::uword core = 0;
        arma::uword nuclear = 0;
    };

    bool GRC_PartialCoreExpectationOracle(
        const arma::cx_mat &operatorCoreNuclear,
        const arma::cx_vec &coreState,
        arma::uword nuclearDimension,
        arma::cx_mat &out)
    {
        out.reset();
        const arma::uword coreDimension=coreState.n_elem;
        if (coreDimension==0 || nuclearDimension==0 ||
            operatorCoreNuclear.n_rows !=
                coreDimension*nuclearDimension ||
            operatorCoreNuclear.n_cols !=
                coreDimension*nuclearDimension)
            return false;

        out.zeros(nuclearDimension,nuclearDimension);
        for (arma::uword i=0;i<coreDimension;++i)
        {
            for (arma::uword j=0;j<coreDimension;++j)
            {
                const arma::cx_double weight=
                    std::conj(coreState(i))*coreState(j);
                if (std::abs(weight)==0.0)
                    continue;

                const arma::uword r0=i*nuclearDimension;
                const arma::uword c0=j*nuclearDimension;
                out += weight*operatorCoreNuclear.submat(
                    r0,c0,
                    r0+nuclearDimension-1,
                    c0+nuclearDimension-1);
            }
        }

        out=0.5*(out+out.t());
        return out.is_finite();
    }

    double GRC_ZfsStateOverlapSlopeError(
        const GRC_ZfsHybridModel &model,
        const RunSection::General::HS::HSOrientation &orientation,
        double hyperfineScale,double fieldT,
        const RunSection::General::Resonance::ResonanceLineSet &exact,
        const RunSection::General::Resonance::ResonanceLineSet &candidate,
        std::size_t count)
    {
        using namespace RunSection::General::Resonance;

        constexpr double minimumStateOverlap=0.80;
        constexpr double stateAmbiguityGap=0.10;
        constexpr double componentFrequencyTolerance=1.0e-8;

        auto system=
            GRC_BuildZfsHybridSystem(
                model,fieldT,hyperfineScale);
        if (system==nullptr)
            return -1.0;

        SpinAPI::SpinSpace fullSpace(*system);
        fullSpace.UseSuperoperatorSpace(false);
        fullSpace.UseFullTensorRotation(true);

        const auto fullPlan=
            GRC_H0Plan({"B0","ZFS","A","NZ"});
        GeneralResonanceHamiltonian fullBuilder(
            fullPlan,fullSpace);

        arma::sp_cx_mat fullH,fullDHdB;
        std::string error;
        if (!fullBuilder.Build(
                orientation,fullH,error) ||
            !fullBuilder.BuildFieldDerivative(
                orientation,{"B0","NZ"},fieldT,
                fullDHdB,error))
            return -1.0;

        arma::vec exactEnergies,exactDEdB;
        arma::cx_mat exactEigenvectors;
        if (!arma::eig_sym(
                exactEnergies,exactEigenvectors,
                arma::cx_mat(fullH)) ||
            !ResonanceFieldJacobian::ResolveDegenerateSubspaces(
                exactEnergies,exactEigenvectors,
                fullDHdB,exactDEdB,error))
            return -1.0;

        HybridNuclearResonancePoint point;
        if (!GRC_BuildZfsHybridPoint(
                model,orientation,hyperfineScale,
                fieldT,point,error))
            return -1.0;

        arma::vec coreEnergies,coreDEdB;
        arma::cx_mat coreEigenvectors;
        if (!arma::eig_sym(
                coreEnergies,coreEigenvectors,
                arma::cx_mat(point.coreHamiltonian)) ||
            !ResonanceFieldJacobian::ResolveDegenerateSubspaces(
                coreEnergies,coreEigenvectors,
                point.coreDHdB,coreDEdB,error))
            return -1.0;

        const arma::uword coreDimension=
            coreEnergies.n_elem;
        const arma::uword nuclearDimension=
            point.hybrid.nuclei.front().nuclearDimension;
        const arma::uword fullDimension=
            coreDimension*nuclearDimension;

        if (nuclearDimension<2 ||
            exactEigenvectors.n_rows!=fullDimension ||
            exactEigenvectors.n_cols!=fullDimension)
            return -1.0;

        std::vector<arma::vec> nuclearEnergies(
            coreDimension);
        std::vector<arma::cx_mat> nuclearEigenvectors(
            coreDimension);

        for (arma::uword a=0;a<coreDimension;++a)
        {
            arma::cx_mat projected;
            if (!GRC_PartialCoreExpectationOracle(
                    point.hybrid.nuclei.front().hyperfineCoreNuclear,
                    coreEigenvectors.col(a),
                    nuclearDimension,projected))
                return -1.0;

            arma::cx_mat effective=
                point.hybrid.nuclei.front().nuclearHamiltonian+projected;
            effective=0.5*(effective+effective.t());

            if (!arma::eig_sym(
                    nuclearEnergies[a],
                    nuclearEigenvectors[a],
                    effective))
                return -1.0;

            arma::vec nuclearDEdB;
            if (!ResonanceFieldJacobian::ResolveDegenerateSubspaces(
                    nuclearEnergies[a],
                    nuclearEigenvectors[a],
                    point.hybrid.nuclei.front().nuclearDHdB,
                    nuclearDEdB,error))
                return -1.0;
        }

        arma::cx_mat productStates(
            fullDimension,fullDimension,arma::fill::zeros);
        std::vector<GRC_ZfsHybridProductLabel> productLabels(
            fullDimension);

        for (arma::uword a=0;a<coreDimension;++a)
        {
            for (arma::uword r=0;r<nuclearDimension;++r)
            {
                const arma::uword j=
                    a*nuclearDimension+r;
                productStates.col(j)=arma::kron(
                    coreEigenvectors.col(a),
                    nuclearEigenvectors[a].col(r));
                productLabels[j]={a,r};
            }
        }

        std::vector<GRC_ZfsHybridProductLabel>
            exactToProduct(fullDimension);
        std::vector<bool> productUsed(
            fullDimension,false);

        for (arma::uword i=0;i<fullDimension;++i)
        {
            double best=-1.0;
            double second=-1.0;
            arma::uword bestIndex=0;

            for (arma::uword j=0;j<fullDimension;++j)
            {
                const double overlap=
                    std::norm(arma::cdot(
                        exactEigenvectors.col(i),
                        productStates.col(j)));

                if (overlap>best)
                {
                    second=best;
                    best=overlap;
                    bestIndex=j;
                }
                else if (overlap>second)
                {
                    second=overlap;
                }
            }

            if (!std::isfinite(best) ||
                best<minimumStateOverlap ||
                (second>=0.0 &&
                 best-second<stateAmbiguityGap) ||
                productUsed[bestIndex])
                return -1.0;

            productUsed[bestIndex]=true;
            exactToProduct[i]=productLabels[bestIndex];
        }

        std::vector<const ResonanceLine *> referenceLines;
        referenceLines.reserve(exact.lines.size());
        for (const auto &line:exact.lines)
        {
            if (GRC_LineWeight(line)>1.0e-16)
                referenceLines.push_back(&line);
        }

        std::sort(
            referenceLines.begin(),referenceLines.end(),
            [](const ResonanceLine *a,
               const ResonanceLine *b)
            {
                return GRC_LineWeight(*a)>
                    GRC_LineWeight(*b);
            });

        if (referenceLines.size()<count)
            return -1.0;
        referenceLines.resize(count);

        std::vector<bool> candidateUsed(
            candidate.lines.size(),false);
        double result=0.0;

        for (const auto *referenceLine:referenceLines)
        {
            if (referenceLine->lower>=exactToProduct.size() ||
                referenceLine->upper>=exactToProduct.size())
                return -1.0;

            const auto lower=
                exactToProduct[referenceLine->lower];
            const auto upper=
                exactToProduct[referenceLine->upper];

            if (lower.core>=upper.core)
                return -1.0;

            const double expectedOmega=
                coreEnergies(upper.core)-
                coreEnergies(lower.core)+
                nuclearEnergies[upper.core](upper.nuclear)-
                nuclearEnergies[lower.core](lower.nuclear);

            double bestDifference=
                std::numeric_limits<double>::infinity();
            double secondDifference=
                std::numeric_limits<double>::infinity();
            std::size_t bestIndex=0;
            bool found=false;

            for (std::size_t j=0;
                 j<candidate.lines.size();++j)
            {
                if (candidateUsed[j])
                    continue;

                const auto &line=candidate.lines[j];
                if (line.lower!=lower.core ||
                    line.upper!=upper.core)
                    continue;

                const double difference=
                    std::abs(line.omega-expectedOmega);
                if (difference<bestDifference)
                {
                    secondDifference=bestDifference;
                    bestDifference=difference;
                    bestIndex=j;
                    found=true;
                }
                else if (difference<secondDifference)
                {
                    secondDifference=difference;
                }
            }

            if (!found ||
                !std::isfinite(bestDifference) ||
                bestDifference>componentFrequencyTolerance ||
                (std::isfinite(secondDifference) &&
                 secondDifference-bestDifference<
                    componentFrequencyTolerance))
                return -1.0;

            candidateUsed[bestIndex]=true;
            result=std::max(
                result,
                std::abs(
                    referenceLine->dOmegaDB-
                    candidate.lines[bestIndex].dOmegaDB));
        }

        return result;
    }

    bool GRC_BuildZfsHybridFiniteDifference(
        const GRC_ZfsHybridModel &model,
        const RunSection::General::HS::HSOrientation &orientation,
        double hyperfineScale,double fieldT,
        double frequencyGHz,double stepT,
        RunSection::General::Resonance::ResonanceLineSet &lines)
    {
        using namespace RunSection::General::Resonance;

        HybridNuclearResonancePointProvider provider =
            [&](double field,
                HybridNuclearResonancePoint &point,
                std::string &error)
            {
                return GRC_BuildZfsHybridPoint(
                    model,orientation,
                    hyperfineScale,field,point,error);
            };

        HybridNuclearResonanceFieldResponseRequest response;
        response.fieldT=fieldT;
        response.fieldStepT=stepT;
        response.minimumCoreStateOverlap=0.85;
        response.minimumNuclearStateOverlap=0.85;
        response.jacobianRelativeTolerance=5.0e-4;
        response.jacobianAbsoluteTolerance=2.0e-4;

        SpectrumRequest request;
        request.microwaveFrequencyGHz=frequencyGHz;
        request.linewidth_mT=0.10;
        request.populationThreshold=1.0e-15;
        request.minimumSlope=1.0e-15;

        std::string error;
        return GRC_GenerateFirstOrderFiniteDifference(
                provider,response,request,lines,error);
    }

    bool GRC_TestHybridZfsFieldResponseS1()
    {
        using namespace RunSection::General::Resonance;

        GRC_ZfsHybridModel model;
        const auto orientation=
            GRC_Orientation(0.37,0.79,-0.26);
        const double frequency=9.5;
        const double fieldT=
            2.0*arma::datum::pi*frequency/
            (GRC_MU_B_OVER_HBAR*model.gz);

        ResonanceLineSet exact,finiteDifference;
        if (!GRC_BuildZfsExactLines(
                model,orientation,0.60,
                fieldT,frequency,exact) ||
            !GRC_BuildZfsHybridFiniteDifference(
                model,orientation,0.60,
                fieldT,frequency,1.0e-4,
                finiteDifference))
            return false;

        if (!finiteDifference.fieldJacobianQualified)
            return false;

        HybridNuclearResonancePoint center;
        std::string error;
        if (!GRC_BuildZfsHybridPoint(
                model,orientation,0.60,
                fieldT,center,error))
            return false;

        SpectrumRequest request;
        request.microwaveFrequencyGHz=frequency;
        request.linewidth_mT=0.10;
        request.populationThreshold=1.0e-15;
        request.minimumSlope=1.0e-15;

        ResonanceLineSet uncorrected;
        if (!GRC_GenerateFirstOrder(
                center.coreHamiltonian,
                center.coreDensity,
                center.coreDHdB,
                center.coreMuX,
                center.coreMuY,
                center.hybrid.nuclei.front(),
                request,uncorrected,error))
            return false;
        if (uncorrected.fieldJacobianQualified)
            return false;

        const double rawError=
            GRC_ZfsStateOverlapSlopeError(
                model,orientation,0.60,fieldT,
                exact,uncorrected,8);
        const double correctedError=
            GRC_ZfsStateOverlapSlopeError(
                model,orientation,0.60,fieldT,
                exact,finiteDifference,8);

        return rawError>0.0 &&
               correctedError>=0.0 &&
               correctedError<0.55*rawError;
    }

    bool GRC_TestHybridZfsFieldResponseA2Scaling()
    {
        GRC_ZfsHybridModel model;
        const auto orientation=
            GRC_Orientation(-0.31,0.68,0.44);
        const double frequency=9.5;
        const double fieldT=
            2.0*arma::datum::pi*frequency/
            (GRC_MU_B_OVER_HBAR*model.gz);

        RunSection::General::Resonance::ResonanceLineSet
            exactStrong,hybridStrong,exactWeak,hybridWeak;

        if (!GRC_BuildZfsExactLines(
                model,orientation,0.60,
                fieldT,frequency,exactStrong) ||
            !GRC_BuildZfsHybridFiniteDifference(
                model,orientation,0.60,
                fieldT,frequency,1.0e-4,
                hybridStrong) ||
            !GRC_BuildZfsExactLines(
                model,orientation,0.30,
                fieldT,frequency,exactWeak) ||
            !GRC_BuildZfsHybridFiniteDifference(
                model,orientation,0.30,
                fieldT,frequency,1.0e-4,
                hybridWeak))
            return false;

        const double strongError=
            GRC_ZfsStateOverlapSlopeError(
                model,orientation,0.60,fieldT,
                exactStrong,hybridStrong,8);
        const double weakError=
            GRC_ZfsStateOverlapSlopeError(
                model,orientation,0.30,fieldT,
                exactWeak,hybridWeak,8);

        if (!(strongError>0.0) ||
            !(weakError>0.0))
            return false;

        const double ratio=strongError/weakError;
        return ratio>2.8 && ratio<5.5;
    }

    bool GRC_TestHybridZfsFieldResponseStepConvergence()
    {
        GRC_ZfsHybridModel model;
        const auto orientation=
            GRC_Orientation(0.25,0.73,-0.48);
        const double frequency=9.5;
        const double fieldT=
            2.0*arma::datum::pi*frequency/
            (GRC_MU_B_OVER_HBAR*model.gz);

        RunSection::General::Resonance::ResonanceLineSet
            coarse,fine;
        if (!GRC_BuildZfsHybridFiniteDifference(
                model,orientation,0.55,
                fieldT,frequency,1.6e-4,coarse) ||
            !GRC_BuildZfsHybridFiniteDifference(
                model,orientation,0.55,
                fieldT,frequency,0.8e-4,fine))
            return false;

        const auto a=GRC_StrongestSlopeLines(coarse,8);
        const auto b=GRC_StrongestSlopeLines(fine,8);
        if (a.size()!=8 || b.size()!=8)
            return false;

        double relative=0.0;
        for (std::size_t i=0;i<8;++i)
        {
            const double scale=
                std::max(1.0,std::abs(b[i].slope));
            relative=std::max(
                relative,
                std::abs(a[i].slope-b[i].slope)/scale);
        }
        return relative<2.0e-4;
    }

    bool GRC_TestHybridZfsFieldResponseS32()
    {
        using namespace RunSection::General::Resonance;

        GRC_ZfsHybridModel model;
        model.electronSpin="3/2";
        model.stateKet="3/2";
        model.d=18.0;
        model.e=4.5;

        const auto orientation=
            GRC_Orientation(-0.22,0.81,0.39);
        const double frequency=94.0;
        const double fieldT=
            2.0*arma::datum::pi*frequency/
            (GRC_MU_B_OVER_HBAR*model.gz);

        ResonanceLineSet exact,corrected;
        if (!GRC_BuildZfsExactLines(
                model,orientation,0.50,
                fieldT,frequency,exact) ||
            !GRC_BuildZfsHybridFiniteDifference(
                model,orientation,0.50,
                fieldT,frequency,2.0e-4,
                corrected))
            return false;
        if (!corrected.fieldJacobianQualified)
            return false;

        HybridNuclearResonancePoint center;
        std::string error;
        if (!GRC_BuildZfsHybridPoint(
                model,orientation,0.50,
                fieldT,center,error))
            return false;

        SpectrumRequest request;
        request.microwaveFrequencyGHz=frequency;
        request.linewidth_mT=0.10;
        request.populationThreshold=1.0e-15;
        request.minimumSlope=1.0e-15;

        ResonanceLineSet uncorrected;
        if (!GRC_GenerateFirstOrder(
                center.coreHamiltonian,
                center.coreDensity,
                center.coreDHdB,
                center.coreMuX,
                center.coreMuY,
                center.hybrid.nuclei.front(),
                request,uncorrected,error))
            return false;

        const double rawError=
            GRC_ZfsStateOverlapSlopeError(
                model,orientation,0.50,fieldT,
                exact,uncorrected,8);
        const double correctedError=
            GRC_ZfsStateOverlapSlopeError(
                model,orientation,0.50,fieldT,
                exact,corrected,8);

        return rawError>0.0 &&
               correctedError>=0.0 &&
               correctedError<rawError;
    }

    struct GRC_ExactCorePromotionModel
    {
        double gx = 1.95;
        double gy = 2.03;
        double gz = 2.18;

        // Strongly coupled nucleus retained in the exact core.
        double axExact = 1.80;
        double ayExact = 2.70;
        double azExact = 4.20;
        double aExactAlpha = 0.34;
        double aExactBeta = 0.59;
        double aExactGamma = -0.23;

        // Second nucleus treated perturbatively.
        double axPert = 0.18;
        double ayPert = 0.31;
        double azPert = 0.52;
        double aPertAlpha = -0.27;
        double aPertBeta = 0.63;
        double aPertGamma = 0.39;
    };

    SpinAPI::system_ptr GRC_BuildExactCorePromotionSystem(
        const GRC_ExactCorePromotionModel &model,
        double fieldT,double perturbativeScale)
    {
        std::ostringstream eprops;
        eprops << std::setprecision(17)
               << "type=electron;spin=1/2;tensor=anisotropic("
               << model.gx << " " << model.gy << " "
               << model.gz << ");";
        auto electron =
            std::make_shared<SpinAPI::Spin>("E",eprops.str());

        auto exactNucleus = std::make_shared<SpinAPI::Spin>(
            "Vexact","type=nucleus;spin=7/2;tensor=isotropic(1);");
        auto perturbativeNucleus = std::make_shared<SpinAPI::Spin>(
            "Vpert","type=nucleus;spin=7/2;tensor=isotropic(1);");

        std::ostringstream bprops;
        bprops << std::setprecision(17)
               << "type=zeeman;spins=E;field=0 0 " << fieldT
               << ";ignoretensors=false;commonprefactor=true;"
                  "prefactor=1.0;";
        auto field =
            std::make_shared<SpinAPI::Interaction>("B0",bprops.str());

        std::ostringstream aExactProps;
        aExactProps << std::setprecision(17)
                    << "type=hyperfine;group1=E;group2=Vexact;"
                       "tensor=anisotropic("
                    << model.axExact << " "
                    << model.ayExact << " "
                    << model.azExact << ");orientation="
                    << model.aExactAlpha << ","
                    << model.aExactBeta << ","
                    << model.aExactGamma
                    << ";ignoretensors=true;commonprefactor=false;"
                       "prefactor=1.0;";
        auto aExact =
            std::make_shared<SpinAPI::Interaction>(
                "Aexact",aExactProps.str());

        std::ostringstream aPertProps;
        aPertProps << std::setprecision(17)
                   << "type=hyperfine;group1=E;group2=Vpert;"
                      "tensor=anisotropic("
                   << perturbativeScale*model.axPert << " "
                   << perturbativeScale*model.ayPert << " "
                   << perturbativeScale*model.azPert
                   << ");orientation="
                   << model.aPertAlpha << ","
                   << model.aPertBeta << ","
                   << model.aPertGamma
                   << ";ignoretensors=true;commonprefactor=false;"
                      "prefactor=1.0;";
        auto aPert =
            std::make_shared<SpinAPI::Interaction>(
                "Apert",aPertProps.str());

        std::ostringstream nzExactProps;
        nzExactProps << std::setprecision(17)
                     << "type=zeeman;spins=Vexact;field=0 0 "
                     << fieldT
                     << ";ignoretensors=true;commonprefactor=false;"
                        "prefactor=0.0703;";
        auto nzExact =
            std::make_shared<SpinAPI::Interaction>(
                "NZexact",nzExactProps.str());

        std::ostringstream nzPertProps;
        nzPertProps << std::setprecision(17)
                    << "type=zeeman;spins=Vpert;field=0 0 "
                    << fieldT
                    << ";ignoretensors=true;commonprefactor=false;"
                       "prefactor=0.0703;";
        auto nzPert =
            std::make_shared<SpinAPI::Interaction>(
                "NZpert",nzPertProps.str());

        auto up = std::make_shared<SpinAPI::State>(
            "Up","spin(E)=|1/2>;");

        auto system =
            std::make_shared<SpinAPI::SpinSystem>(
                "ExactCorePromotion");
        system->Add(electron);
        system->Add(exactNucleus);
        system->Add(perturbativeNucleus);
        system->Add(field);
        system->Add(aExact);
        system->Add(aPert);
        system->Add(nzExact);
        system->Add(nzPert);
        system->Add(up);
        system->SetProperties(
            std::make_shared<MSDParser::ObjectParser>(
                "properties","initialstate=Up;"));

        if (!system->ValidateInteractions().empty())
            return nullptr;
        if (!up->ParseFromSystem(*system))
            return nullptr;
        return system;
    }

    bool GRC_BuildExactCorePromotionPoint(
        const GRC_ExactCorePromotionModel &model,
        const RunSection::General::HS::HSOrientation &orientation,
        double perturbativeScale,double fieldT,bool swapCoreOrder,
        RunSection::General::Resonance::HybridNuclearResonancePoint &point,
        std::string &error)
    {
        using namespace RunSection::General::Resonance;

        auto system =
            GRC_BuildExactCorePromotionSystem(
                model,fieldT,perturbativeScale);
        if (system == nullptr)
        {
            error =
                "failed to build exact-core promotion system";
            return false;
        }

        auto electron = system->spins_find("E");
        auto exactNucleus = system->spins_find("Vexact");
        auto perturbativeNucleus = system->spins_find("Vpert");
        auto b0 = system->interactions_find("B0");
        auto aExact = system->interactions_find("Aexact");
        auto aPert = system->interactions_find("Apert");
        auto nzExact = system->interactions_find("NZexact");
        auto nzPert = system->interactions_find("NZpert");

        if (electron == nullptr ||
            exactNucleus == nullptr ||
            perturbativeNucleus == nullptr ||
            b0 == nullptr ||
            aExact == nullptr ||
            aPert == nullptr ||
            nzExact == nullptr ||
            nzPert == nullptr)
        {
            error = "missing exact-core promotion object";
            return false;
        }

        std::vector<SpinAPI::spin_ptr> coreSpins;
        if (swapCoreOrder)
            coreSpins = {exactNucleus,electron};
        else
            coreSpins = {electron,exactNucleus};

        SpinAPI::SpinSpace coreSpace(coreSpins);
        coreSpace.Add(b0);
        coreSpace.Add(aExact);
        coreSpace.Add(nzExact);
        coreSpace.UseSuperoperatorSpace(false);
        coreSpace.UseFullTensorRotation(true);

        const auto corePlan =
            GRC_H0Plan({"B0","Aexact","NZexact"});
        GeneralResonanceHamiltonian coreBuilder(
            corePlan,coreSpace);

        if (!coreBuilder.Build(
                orientation,point.coreHamiltonian,error) ||
            !coreBuilder.BuildFieldDerivative(
                orientation,{"B0","NZexact"},fieldT,
                point.coreDHdB,error) ||
            !GRC_Density(
                system,coreSpace,point.coreDensity) ||
            !GRC_TransverseOperators(
                system,coreSpace,orientation,
                point.coreMuX,point.coreMuY))
            return false;

        // Preserve the exact-core basis order and append the perturbative
        // nucleus as the final Kronecker factor required by the R2A/R2C
        // partial-core projection contract.
        std::vector<SpinAPI::spin_ptr> pairSpins = coreSpins;
        pairSpins.push_back(perturbativeNucleus);

        SpinAPI::SpinSpace pairSpace(pairSpins);
        pairSpace.UseSuperoperatorSpace(false);
        pairSpace.UseFullTensorRotation(true);

        arma::mat rotation = orientation.frameToLab;
        arma::sp_cx_mat hyperfinePair;
        if (!pairSpace.InteractionOperatorRotatedZYZ(
                aPert,rotation,hyperfinePair))
        {
            error =
                "failed to build exact-core/perturbative hyperfine operator";
            return false;
        }

        SpinAPI::SpinSpace nuclearSpace(perturbativeNucleus);
        nuclearSpace.UseSuperoperatorSpace(false);
        nuclearSpace.UseFullTensorRotation(true);

        arma::sp_cx_mat nuclearH;
        if (!nuclearSpace.InteractionOperatorRotatedZYZ(
                nzPert,rotation,nuclearH))
        {
            error =
                "failed to build perturbative nuclear Zeeman operator";
            return false;
        }

        point.hybrid.nuclei.resize(1);
        point.hybrid.nuclei.front().hyperfineCoreNuclear =
            arma::cx_mat(hyperfinePair);
        point.hybrid.nuclei.front().nuclearHamiltonian =
            arma::cx_mat(nuclearH);
        point.hybrid.nuclei.front().nuclearDHdB =
            nuclearH/fieldT;
        point.hybrid.nuclei.front().nuclearDimension =
            static_cast<arma::uword>(
                perturbativeNucleus->Multiplicity());
        point.hybrid.nuclei.front().overlapThreshold = 1.0e-14;

        // The exact core contains a strongly coupled nucleus, so its
        // eigenvectors are field-dependent even for S=1/2.
        point.hybrid.nuclei.front().fieldIndependentProjection = false;

        return true;
    }

    bool GRC_BuildExactCorePromotionExactLines(
        const GRC_ExactCorePromotionModel &model,
        const RunSection::General::HS::HSOrientation &orientation,
        double perturbativeScale,double fieldT,
        double frequencyGHz,
        RunSection::General::Resonance::ResonanceLineSet &lines)
    {
        using namespace RunSection::General::Resonance;

        auto system =
            GRC_BuildExactCorePromotionSystem(
                model,fieldT,perturbativeScale);
        if (system == nullptr)
            return false;

        SpinAPI::SpinSpace fullSpace(*system);
        fullSpace.UseSuperoperatorSpace(false);
        fullSpace.UseFullTensorRotation(true);

        const auto fullPlan =
            GRC_H0Plan({
                "B0","Aexact","Apert","NZexact","NZpert"});
        GeneralResonanceHamiltonian fullBuilder(
            fullPlan,fullSpace);

        arma::sp_cx_mat H,dHdB;
        arma::cx_mat rho,muX,muY;
        std::string error;

        if (!fullBuilder.Build(orientation,H,error) ||
            !fullBuilder.BuildFieldDerivative(
                orientation,
                {"B0","NZexact","NZpert"},
                fieldT,dHdB,error) ||
            !GRC_Density(system,fullSpace,rho) ||
            !GRC_TransverseOperators(
                system,fullSpace,orientation,muX,muY))
            return false;

        SpectrumRequest request;
        request.microwaveFrequencyGHz=frequencyGHz;
        request.linewidth_mT=0.10;
        request.populationThreshold=1.0e-15;
        request.minimumSlope=1.0e-15;

        return ExactResonanceSolver::Generate(
            H,rho,dHdB,muX,muY,request,lines,error);
    }

    bool GRC_BuildExactCorePromotionHybridLines(
        const GRC_ExactCorePromotionModel &model,
        const RunSection::General::HS::HSOrientation &orientation,
        double perturbativeScale,double fieldT,
        double frequencyGHz,double stepT,bool swapCoreOrder,
        RunSection::General::Resonance::ResonanceLineSet &lines)
    {
        using namespace RunSection::General::Resonance;

        HybridNuclearResonancePointProvider provider =
            [&](double field,
                HybridNuclearResonancePoint &point,
                std::string &error)
            {
                return GRC_BuildExactCorePromotionPoint(
                    model,orientation,perturbativeScale,
                    field,swapCoreOrder,point,error);
            };

        HybridNuclearResonanceFieldResponseRequest response;
        response.fieldT=fieldT;
        response.fieldStepT=stepT;
        response.minimumCoreStateOverlap=0.85;
        response.minimumNuclearStateOverlap=0.85;
        response.jacobianRelativeTolerance=5.0e-4;
        response.jacobianAbsoluteTolerance=2.0e-4;

        SpectrumRequest request;
        request.microwaveFrequencyGHz=frequencyGHz;
        request.linewidth_mT=0.10;
        request.populationThreshold=1.0e-15;
        request.minimumSlope=1.0e-15;

        std::string error;
        return GRC_GenerateFirstOrderFiniteDifference(
                provider,response,request,lines,error);
    }

    bool GRC_TestHybridExactCorePromotionZeroPerturbativeParity()
    {
        using namespace RunSection::General::Resonance;

        GRC_ExactCorePromotionModel model;
        const auto orientation =
            GRC_Orientation(0.28,0.67,-0.36);
        const double frequency = 9.5;
        const double fieldT =
            2.0*arma::datum::pi*frequency/
            (GRC_MU_B_OVER_HBAR*model.gz);

        HybridNuclearResonancePoint center;
        std::string error;
        if (!GRC_BuildExactCorePromotionPoint(
                model,orientation,0.0,fieldT,false,
                center,error))
            return false;

        if (center.coreHamiltonian.n_rows != 16 ||
            center.coreHamiltonian.n_cols != 16 ||
            center.hybrid.nuclei.front().nuclearDimension != 8 ||
            center.hybrid.nuclei.front().hyperfineCoreNuclear.n_rows != 128 ||
            center.hybrid.nuclei.front().hyperfineCoreNuclear.n_cols != 128)
            return false;

        ResonanceLineSet exact,hybrid;
        if (!GRC_BuildExactCorePromotionExactLines(
                model,orientation,0.0,fieldT,
                frequency,exact) ||
            !GRC_BuildExactCorePromotionHybridLines(
                model,orientation,0.0,fieldT,
                frequency,5.0e-5,false,hybrid))
            return false;

        if (!hybrid.fieldJacobianQualified)
            return false;

        SpectrumRequest request;
        request.microwaveFrequencyGHz=frequency;
        request.linewidth_mT=3.0;
        request.populationThreshold=1.0e-15;
        request.minimumSlope=1.0e-15;

        SpectrumPoint exactPoint,hybridPoint;
        if (!ResonanceSpectrumEvaluator::Evaluate(
                exact,request,exactPoint,error) ||
            !ResonanceSpectrumEvaluator::Evaluate(
                hybrid,request,hybridPoint,error))
            return false;

        const double scale=std::max({
            1.0,
            std::abs(exactPoint.totalX),
            std::abs(exactPoint.totalY),
            std::abs(exactPoint.totalPerpendicular)
        });

        if (std::abs(exactPoint.totalPerpendicular) < 1.0e-12)
            return false;

        return
            std::abs(exactPoint.totalX-hybridPoint.totalX)
                <= 2.0e-6*scale &&
            std::abs(exactPoint.totalY-hybridPoint.totalY)
                <= 2.0e-6*scale &&
            std::abs(
                exactPoint.totalPerpendicular-
                hybridPoint.totalPerpendicular)
                <= 2.0e-6*scale;
    }

    bool GRC_TestHybridExactCorePromotionBasisOrderInvariant()
    {
        using namespace RunSection::General::Resonance;

        GRC_ExactCorePromotionModel model;
        const auto orientation =
            GRC_Orientation(-0.24,0.74,0.43);
        const double frequency = 9.5;
        const double fieldT =
            2.0*arma::datum::pi*frequency/
            (GRC_MU_B_OVER_HBAR*model.gz);

        ResonanceLineSet canonical,swapped;
        if (!GRC_BuildExactCorePromotionHybridLines(
                model,orientation,0.45,fieldT,
                frequency,7.5e-5,false,canonical) ||
            !GRC_BuildExactCorePromotionHybridLines(
                model,orientation,0.45,fieldT,
                frequency,7.5e-5,true,swapped))
            return false;

        if (!canonical.fieldJacobianQualified ||
            !swapped.fieldJacobianQualified)
            return false;

        SpectrumRequest request;
        request.microwaveFrequencyGHz=frequency;
        request.linewidth_mT=2.0;
        request.populationThreshold=1.0e-15;
        request.minimumSlope=1.0e-15;

        SpectrumPoint a,b;
        std::string error;
        if (!ResonanceSpectrumEvaluator::Evaluate(
                canonical,request,a,error) ||
            !ResonanceSpectrumEvaluator::Evaluate(
                swapped,request,b,error))
            return false;

        const double scale=std::max({
            1.0,
            std::abs(a.totalX),
            std::abs(a.totalY),
            std::abs(a.totalPerpendicular)
        });

        return
            std::abs(a.totalX-b.totalX) <= 1.0e-9*scale &&
            std::abs(a.totalY-b.totalY) <= 1.0e-9*scale &&
            std::abs(a.totalPerpendicular-b.totalPerpendicular)
                <= 1.0e-9*scale;
    }

    struct GRC_ExactCoreHybridProductLabel
    {
        arma::uword core = 0;
        arma::uword nuclear = 0;
    };

    double GRC_ExactCoreStateOverlapSlopeError(
        const GRC_ExactCorePromotionModel &model,
        const RunSection::General::HS::HSOrientation &orientation,
        double perturbativeScale,double fieldT,
        const RunSection::General::Resonance::ResonanceLineSet &exact,
        const RunSection::General::Resonance::ResonanceLineSet &candidate,
        std::size_t count)
    {
        using namespace RunSection::General::Resonance;

        // Validation-only oracle for the nondegenerate synthetic R2D gate.
        // Exact strong-line intensity selects which exact transitions matter,
        // but exact<->hybrid identity is determined solely from wavefunctions.
        constexpr double minimumStateOverlap=0.80;
        constexpr double stateAmbiguityGap=0.10;
        constexpr double componentFrequencyTolerance=1.0e-8;

        auto system=
            GRC_BuildExactCorePromotionSystem(
                model,fieldT,perturbativeScale);
        if (system==nullptr)
            return -1.0;

        SpinAPI::SpinSpace fullSpace(*system);
        fullSpace.UseSuperoperatorSpace(false);
        fullSpace.UseFullTensorRotation(true);

        const auto fullPlan=
            GRC_H0Plan({
                "B0","Aexact","Apert","NZexact","NZpert"});
        GeneralResonanceHamiltonian fullBuilder(
            fullPlan,fullSpace);

        arma::sp_cx_mat fullH,fullDHdB;
        std::string error;
        if (!fullBuilder.Build(
                orientation,fullH,error) ||
            !fullBuilder.BuildFieldDerivative(
                orientation,
                {"B0","NZexact","NZpert"},
                fieldT,fullDHdB,error))
            return -1.0;

        arma::vec exactEnergies,exactDEdB;
        arma::cx_mat exactEigenvectors;
        if (!arma::eig_sym(
                exactEnergies,exactEigenvectors,
                arma::cx_mat(fullH)) ||
            !ResonanceFieldJacobian::ResolveDegenerateSubspaces(
                exactEnergies,exactEigenvectors,
                fullDHdB,exactDEdB,error))
            return -1.0;

        HybridNuclearResonancePoint point;
        if (!GRC_BuildExactCorePromotionPoint(
                model,orientation,perturbativeScale,
                fieldT,false,point,error))
            return -1.0;

        arma::vec coreEnergies,coreDEdB;
        arma::cx_mat coreEigenvectors;
        if (!arma::eig_sym(
                coreEnergies,coreEigenvectors,
                arma::cx_mat(point.coreHamiltonian)) ||
            !ResonanceFieldJacobian::ResolveDegenerateSubspaces(
                coreEnergies,coreEigenvectors,
                point.coreDHdB,coreDEdB,error))
            return -1.0;

        const arma::uword coreDimension=
            coreEnergies.n_elem;
        const arma::uword nuclearDimension=
            point.hybrid.nuclei.front().nuclearDimension;
        const arma::uword fullDimension=
            coreDimension*nuclearDimension;

        if (nuclearDimension<2 ||
            exactEigenvectors.n_rows!=fullDimension ||
            exactEigenvectors.n_cols!=fullDimension)
            return -1.0;

        std::vector<arma::vec> nuclearEnergies(
            coreDimension);
        std::vector<arma::cx_mat> nuclearEigenvectors(
            coreDimension);

        for (arma::uword a=0;a<coreDimension;++a)
        {
            arma::cx_mat projected;
            if (!GRC_PartialCoreExpectationOracle(
                    point.hybrid.nuclei.front().hyperfineCoreNuclear,
                    coreEigenvectors.col(a),
                    nuclearDimension,projected))
                return -1.0;

            arma::cx_mat effective=
                point.hybrid.nuclei.front().nuclearHamiltonian+projected;
            effective=0.5*(effective+effective.t());

            if (!arma::eig_sym(
                    nuclearEnergies[a],
                    nuclearEigenvectors[a],
                    effective))
                return -1.0;

            arma::vec nuclearDEdB;
            if (!ResonanceFieldJacobian::ResolveDegenerateSubspaces(
                    nuclearEnergies[a],
                    nuclearEigenvectors[a],
                    point.hybrid.nuclei.front().nuclearDHdB,
                    nuclearDEdB,error))
                return -1.0;
        }

        arma::cx_mat productStates(
            fullDimension,fullDimension,arma::fill::zeros);
        std::vector<GRC_ExactCoreHybridProductLabel>
            productLabels(fullDimension);

        for (arma::uword a=0;a<coreDimension;++a)
        {
            for (arma::uword r=0;r<nuclearDimension;++r)
            {
                const arma::uword j=
                    a*nuclearDimension+r;
                productStates.col(j)=arma::kron(
                    coreEigenvectors.col(a),
                    nuclearEigenvectors[a].col(r));
                productLabels[j]={a,r};
            }
        }

        std::vector<GRC_ExactCoreHybridProductLabel>
            exactToProduct(fullDimension);
        std::vector<bool> productUsed(
            fullDimension,false);

        for (arma::uword i=0;i<fullDimension;++i)
        {
            double best=-1.0;
            double second=-1.0;
            arma::uword bestIndex=0;

            for (arma::uword j=0;j<fullDimension;++j)
            {
                const double overlap=
                    std::norm(arma::cdot(
                        exactEigenvectors.col(i),
                        productStates.col(j)));

                if (overlap>best)
                {
                    second=best;
                    best=overlap;
                    bestIndex=j;
                }
                else if (overlap>second)
                {
                    second=overlap;
                }
            }

            if (!std::isfinite(best) ||
                best<minimumStateOverlap ||
                (second>=0.0 &&
                 best-second<stateAmbiguityGap) ||
                productUsed[bestIndex])
                return -1.0;

            productUsed[bestIndex]=true;
            exactToProduct[i]=productLabels[bestIndex];
        }

        std::vector<const ResonanceLine *> referenceLines;
        referenceLines.reserve(exact.lines.size());
        for (const auto &line:exact.lines)
        {
            if (GRC_LineWeight(line)>1.0e-16)
                referenceLines.push_back(&line);
        }

        std::sort(
            referenceLines.begin(),referenceLines.end(),
            [](const ResonanceLine *a,
               const ResonanceLine *b)
            {
                return GRC_LineWeight(*a)>
                    GRC_LineWeight(*b);
            });

        if (referenceLines.size()<count)
            return -1.0;
        referenceLines.resize(count);

        std::vector<bool> candidateUsed(
            candidate.lines.size(),false);
        double result=0.0;

        for (const auto *referenceLine:referenceLines)
        {
            if (referenceLine->lower>=exactToProduct.size() ||
                referenceLine->upper>=exactToProduct.size())
                return -1.0;

            const auto lower=
                exactToProduct[referenceLine->lower];
            const auto upper=
                exactToProduct[referenceLine->upper];

            if (lower.core>=upper.core)
                return -1.0;

            const double expectedOmega=
                coreEnergies(upper.core)-
                coreEnergies(lower.core)+
                nuclearEnergies[upper.core](upper.nuclear)-
                nuclearEnergies[lower.core](lower.nuclear);

            double bestDifference=
                std::numeric_limits<double>::infinity();
            double secondDifference=
                std::numeric_limits<double>::infinity();
            std::size_t bestIndex=0;
            bool found=false;

            for (std::size_t j=0;
                 j<candidate.lines.size();++j)
            {
                if (candidateUsed[j])
                    continue;

                const auto &line=candidate.lines[j];
                if (line.lower!=lower.core ||
                    line.upper!=upper.core)
                    continue;

                const double difference=
                    std::abs(line.omega-expectedOmega);
                if (difference<bestDifference)
                {
                    secondDifference=bestDifference;
                    bestDifference=difference;
                    bestIndex=j;
                    found=true;
                }
                else if (difference<secondDifference)
                {
                    secondDifference=difference;
                }
            }

            if (!found ||
                !std::isfinite(bestDifference) ||
                bestDifference>componentFrequencyTolerance ||
                (std::isfinite(secondDifference) &&
                 secondDifference-bestDifference<
                    componentFrequencyTolerance))
                return -1.0;

            candidateUsed[bestIndex]=true;
            result=std::max(
                result,
                std::abs(
                    referenceLine->dOmegaDB-
                    candidate.lines[bestIndex].dOmegaDB));
        }

        return result;
    }

    bool GRC_TestHybridExactCorePromotionFieldResponse()
    {
        using namespace RunSection::General::Resonance;

        GRC_ExactCorePromotionModel model;
        const auto orientation =
            GRC_Orientation(0.35,0.71,-0.31);
        const double frequency = 9.5;
        const double fieldT =
            2.0*arma::datum::pi*frequency/
            (GRC_MU_B_OVER_HBAR*model.gz);

        ResonanceLineSet exact,corrected;
        if (!GRC_BuildExactCorePromotionExactLines(
                model,orientation,0.60,fieldT,
                frequency,exact) ||
            !GRC_BuildExactCorePromotionHybridLines(
                model,orientation,0.60,fieldT,
                frequency,7.5e-5,false,corrected))
            return false;

        HybridNuclearResonancePoint center;
        std::string error;
        if (!GRC_BuildExactCorePromotionPoint(
                model,orientation,0.60,fieldT,false,
                center,error))
            return false;

        SpectrumRequest request;
        request.microwaveFrequencyGHz=frequency;
        request.linewidth_mT=0.10;
        request.populationThreshold=1.0e-15;
        request.minimumSlope=1.0e-15;

        ResonanceLineSet incomplete;
        if (!GRC_GenerateFirstOrder(
                center.coreHamiltonian,
                center.coreDensity,
                center.coreDHdB,
                center.coreMuX,
                center.coreMuY,
                center.hybrid.nuclei.front(),
                request,incomplete,error))
            return false;

        if (incomplete.fieldJacobianQualified ||
            !corrected.fieldJacobianQualified)
            return false;

        const double rawError =
            GRC_ExactCoreStateOverlapSlopeError(
                model,orientation,0.60,fieldT,
                exact,incomplete,8);
        const double correctedError =
            GRC_ExactCoreStateOverlapSlopeError(
                model,orientation,0.60,fieldT,
                exact,corrected,8);

        return rawError>0.0 &&
               correctedError>=0.0 &&
               correctedError<0.50*rawError;
    }

    bool GRC_TestHybridExactCorePromotionA2Scaling()
    {
        GRC_ExactCorePromotionModel model;
        const auto orientation =
            GRC_Orientation(-0.32,0.69,0.38);
        const double frequency = 9.5;
        const double fieldT =
            2.0*arma::datum::pi*frequency/
            (GRC_MU_B_OVER_HBAR*model.gz);

        RunSection::General::Resonance::ResonanceLineSet
            exactStrong,hybridStrong,exactWeak,hybridWeak;

        if (!GRC_BuildExactCorePromotionExactLines(
                model,orientation,0.60,fieldT,
                frequency,exactStrong) ||
            !GRC_BuildExactCorePromotionHybridLines(
                model,orientation,0.60,fieldT,
                frequency,7.5e-5,false,hybridStrong) ||
            !GRC_BuildExactCorePromotionExactLines(
                model,orientation,0.30,fieldT,
                frequency,exactWeak) ||
            !GRC_BuildExactCorePromotionHybridLines(
                model,orientation,0.30,fieldT,
                frequency,7.5e-5,false,hybridWeak))
            return false;

        const double strongError =
            GRC_ExactCoreStateOverlapSlopeError(
                model,orientation,0.60,fieldT,
                exactStrong,hybridStrong,8);
        const double weakError =
            GRC_ExactCoreStateOverlapSlopeError(
                model,orientation,0.30,fieldT,
                exactWeak,hybridWeak,8);

        if (!(strongError>0.0) ||
            !(weakError>0.0))
            return false;

        const double ratio=strongError/weakError;
        return ratio>2.8 && ratio<5.5;
    }

    bool GRC_R2GA_LineFixedFieldNear(
        const RunSection::General::Resonance::ResonanceLine &a,
        const RunSection::General::Resonance::ResonanceLine &b,
        double tolerance)
    {
        if (a.lower!=b.lower || a.upper!=b.upper)
            return false;

        const double scale=std::max({
            1.0,
            std::abs(a.omega),std::abs(b.omega),
            std::abs(a.populationDifference),
            std::abs(b.populationDifference),
            std::abs(a.moment.x),std::abs(b.moment.x),
            std::abs(a.moment.y),std::abs(b.moment.y),
            std::abs(a.moment.perpendicular),
            std::abs(b.moment.perpendicular)
        });

        return
            std::abs(a.omega-b.omega)<=tolerance*scale &&
            std::abs(
                a.populationDifference-
                b.populationDifference)<=tolerance*scale &&
            std::abs(
                a.moment.x-b.moment.x)<=tolerance*scale &&
            std::abs(
                a.moment.y-b.moment.y)<=tolerance*scale &&
            std::abs(
                a.moment.perpendicular-
                b.moment.perpendicular)<=tolerance*scale;
    }

    std::vector<RunSection::General::Resonance::ResonanceLine>
    GRC_R2GA_SortedFixedFieldLines(
        const RunSection::General::Resonance::ResonanceLineSet &set)
    {
        auto lines=set.lines;
        std::sort(
            lines.begin(),lines.end(),
            [](const auto &a,const auto &b)
            {
                if (a.lower!=b.lower)
                    return a.lower<b.lower;
                if (a.upper!=b.upper)
                    return a.upper<b.upper;
                if (a.omega!=b.omega)
                    return a.omega<b.omega;
                if (a.moment.perpendicular!=
                    b.moment.perpendicular)
                    return
                        a.moment.perpendicular<
                        b.moment.perpendicular;
                return
                    a.populationDifference<
                    b.populationDifference;
            });
        return lines;
    }

    bool GRC_TestR2GAMultiNucleusN1Parity()
    {
        using namespace RunSection::General::Resonance;

        GRC_ZfsHybridModel model;
        const auto orientation=
            GRC_Orientation(0.31,0.69,-0.27);
        const double frequency=9.5;
        const double fieldT=
            2.0*arma::datum::pi*frequency/
            (GRC_MU_B_OVER_HBAR*model.gz);

        HybridNuclearResonancePoint point;
        std::string error;
        if (!GRC_BuildZfsHybridPoint(
                model,orientation,0.47,fieldT,
                point,error))
            return false;

        SpectrumRequest request;
        request.microwaveFrequencyGHz=frequency;
        request.linewidth_mT=0.10;
        request.populationThreshold=1.0e-15;
        request.minimumSlope=1.0e-15;

        ResonanceLineSet one,multi;
        if (!GRC_GenerateFirstOrder(
                point.coreHamiltonian,
                point.coreDensity,
                point.coreDHdB,
                point.coreMuX,
                point.coreMuY,
                point.hybrid.nuclei.front(),
                request,one,error))
            return false;

        HybridNuclearResonanceRequest multiRequest;
        multiRequest.nuclei={point.hybrid.nuclei.front()};
        HybridNuclearResonanceReport report;
        if (!GRC_GenerateFirstOrder(
                    point.coreHamiltonian,
                    point.coreDensity,
                    point.coreDHdB,
                    point.coreMuX,
                    point.coreMuY,
                    multiRequest,request,
                    multi,report,error))
            return false;

        if (multi.fieldJacobianQualified ||
            report.nucleusCount!=1 ||
            report.productNuclearDimension!=8 ||
            report.largestDiagonalizedNuclearDimension!=8 ||
            report.maximumDiscardedNuclearWeightFraction>
                1.0e-12)
            return false;

        const auto a=GRC_R2GA_SortedFixedFieldLines(one);
        const auto b=GRC_R2GA_SortedFixedFieldLines(multi);
        if (a.size()!=b.size())
            return false;

        for (std::size_t i=0;i<a.size();++i)
        {
            if (!GRC_R2GA_LineFixedFieldNear(
                    a[i],b[i],1.0e-13))
                return false;
        }
        return true;
    }

    bool GRC_TestR2GAMultiNucleusPermutationInvariant()
    {
        using namespace RunSection::General::Resonance;

        GRC_ZfsHybridModel model;
        const auto orientation=
            GRC_Orientation(-0.24,0.74,0.41);
        const double frequency=9.5;
        const double fieldT=
            2.0*arma::datum::pi*frequency/
            (GRC_MU_B_OVER_HBAR*model.gz);

        HybridNuclearResonancePoint a,b;
        std::string error;
        if (!GRC_BuildZfsHybridPoint(
                model,orientation,0.41,fieldT,a,error) ||
            !GRC_BuildZfsHybridPoint(
                model,orientation,0.63,fieldT,b,error))
            return false;

        SpectrumRequest request;
        request.microwaveFrequencyGHz=frequency;
        request.linewidth_mT=0.10;
        request.populationThreshold=1.0e-15;

        HybridNuclearResonanceRequest ab,ba;
        ab.nuclei={a.hybrid.nuclei.front(),b.hybrid.nuclei.front()};
        ba.nuclei={b.hybrid.nuclei.front(),a.hybrid.nuclei.front()};

        ResonanceLineSet linesAB,linesBA;
        HybridNuclearResonanceReport reportAB,reportBA;
        if (!GRC_GenerateFirstOrder(
                    a.coreHamiltonian,a.coreDensity,
                    a.coreDHdB,a.coreMuX,a.coreMuY,
                    ab,request,linesAB,reportAB,error) ||
            !GRC_GenerateFirstOrder(
                    a.coreHamiltonian,a.coreDensity,
                    a.coreDHdB,a.coreMuX,a.coreMuY,
                    ba,request,linesBA,reportBA,error))
            return false;

        if (reportAB.productNuclearDimension!=64 ||
            reportBA.productNuclearDimension!=64 ||
            reportAB.largestDiagonalizedNuclearDimension!=8 ||
            reportBA.largestDiagonalizedNuclearDimension!=8 ||
            reportAB.maximumDiscardedNuclearWeightFraction>
                1.0e-11 ||
            reportBA.maximumDiscardedNuclearWeightFraction>
                1.0e-11)
            return false;

        const auto x=GRC_R2GA_SortedFixedFieldLines(linesAB);
        const auto y=GRC_R2GA_SortedFixedFieldLines(linesBA);
        if (x.size()!=y.size() || x.empty())
            return false;

        for (std::size_t i=0;i<x.size();++i)
        {
            if (!GRC_R2GA_LineFixedFieldNear(
                    x[i],y[i],2.0e-12))
                return false;
        }
        return true;
    }

    bool GRC_TestR2GAMultiNucleusSpectatorAndWeightConservation()
    {
        using namespace RunSection::General::Resonance;

        GRC_ZfsHybridModel model;
        const auto orientation=
            GRC_Orientation(0.22,0.67,-0.39);
        const double frequency=9.5;
        const double fieldT=
            2.0*arma::datum::pi*frequency/
            (GRC_MU_B_OVER_HBAR*model.gz);

        HybridNuclearResonancePoint active;
        std::string error;
        if (!GRC_BuildZfsHybridPoint(
                model,orientation,0.52,fieldT,
                active,error))
            return false;

        HybridNuclearResonanceNucleus spectator=
            active.hybrid.nuclei.front();
        spectator.hyperfineCoreNuclear.zeros(
            active.hybrid.nuclei.front().hyperfineCoreNuclear.n_rows,
            active.hybrid.nuclei.front().hyperfineCoreNuclear.n_cols);

        SpectrumRequest request;
        request.microwaveFrequencyGHz=frequency;
        request.linewidth_mT=0.10;
        request.populationThreshold=1.0e-15;

        HybridNuclearResonanceRequest one,two;
        one.nuclei={active.hybrid.nuclei.front()};
        two.nuclei={active.hybrid.nuclei.front(),spectator};
        one.mergeFrequencyToleranceRadNs=1.0e-11;
        two.mergeFrequencyToleranceRadNs=1.0e-11;

        ResonanceLineSet oneLines,twoLines;
        HybridNuclearResonanceReport oneReport,twoReport;
        if (!GRC_GenerateFirstOrder(
                    active.coreHamiltonian,
                    active.coreDensity,
                    active.coreDHdB,
                    active.coreMuX,
                    active.coreMuY,
                    one,request,
                    oneLines,oneReport,error) ||
            !GRC_GenerateFirstOrder(
                    active.coreHamiltonian,
                    active.coreDensity,
                    active.coreDHdB,
                    active.coreMuX,
                    active.coreMuY,
                    two,request,
                    twoLines,twoReport,error))
            return false;

        if (!twoReport.mergingApplied ||
            oneReport.maximumDiscardedNuclearWeightFraction>
                1.0e-11 ||
            twoReport.maximumDiscardedNuclearWeightFraction>
                1.0e-11)
            return false;

        const auto a=GRC_R2GA_SortedFixedFieldLines(oneLines);
        const auto b=GRC_R2GA_SortedFixedFieldLines(twoLines);
        if (a.size()!=b.size() || a.empty())
            return false;

        for (std::size_t i=0;i<a.size();++i)
        {
            if (a[i].lower!=b[i].lower ||
                a[i].upper!=b[i].upper)
                return false;

            const double scale=std::max({
                1.0,std::abs(a[i].omega),
                std::abs(b[i].omega)
            });
            if (std::abs(a[i].omega-b[i].omega)>
                2.0e-11*scale)
                return false;

            const double ax=
                a[i].populationDifference*a[i].moment.x;
            const double bx=
                b[i].populationDifference*b[i].moment.x;
            const double ay=
                a[i].populationDifference*a[i].moment.y;
            const double by=
                b[i].populationDifference*b[i].moment.y;
            const double ap=
                a[i].populationDifference*
                a[i].moment.perpendicular;
            const double bp=
                b[i].populationDifference*
                b[i].moment.perpendicular;
            const double wscale=std::max({
                1.0,std::abs(ax),std::abs(bx),
                std::abs(ay),std::abs(by),
                std::abs(ap),std::abs(bp)
            });
            if (std::abs(ax-bx)>2.0e-11*wscale ||
                std::abs(ay-by)>2.0e-11*wscale ||
                std::abs(ap-bp)>2.0e-11*wscale)
                return false;
        }

        return true;
    }

    bool GRC_TestR2GAMultiNucleusControlsFailClosed()
    {
        using namespace RunSection::General::Resonance;

        GRC_ZfsHybridModel model;
        const auto orientation=
            GRC_Orientation(-0.35,0.71,0.28);
        const double frequency=9.5;
        const double fieldT=
            2.0*arma::datum::pi*frequency/
            (GRC_MU_B_OVER_HBAR*model.gz);

        HybridNuclearResonancePoint a,b;
        std::string error;
        if (!GRC_BuildZfsHybridPoint(
                model,orientation,0.44,fieldT,a,error) ||
            !GRC_BuildZfsHybridPoint(
                model,orientation,0.61,fieldT,b,error))
            return false;

        SpectrumRequest request;
        request.microwaveFrequencyGHz=frequency;
        request.linewidth_mT=0.10;
        request.populationThreshold=1.0e-15;

        HybridNuclearResonanceRequest capped;
        capped.nuclei={a.hybrid.nuclei.front(),b.hybrid.nuclei.front()};
        capped.maximumComponentsPerCoreTransition=4;

        ResonanceLineSet lines;
        HybridNuclearResonanceReport report;
        if (GRC_GenerateFirstOrder(
                    a.coreHamiltonian,a.coreDensity,
                    a.coreDHdB,a.coreMuX,a.coreMuY,
                    capped,request,lines,report,error) ||
            error !=
                "independent multi-nucleus component cap exceeded")
            return false;

        HybridNuclearResonanceRequest pruned;
        pruned.nuclei={a.hybrid.nuclei.front(),b.hybrid.nuclei.front()};
        pruned.minimumCumulativeOverlapWeight=0.999999;

        if (!GRC_GenerateFirstOrder(
                    a.coreHamiltonian,a.coreDensity,
                    a.coreDHdB,a.coreMuX,a.coreMuY,
                    pruned,request,lines,report,error))
            return false;

        return
            report.pruningApplied &&
            report.maximumDiscardedNuclearWeightFraction>0.0 &&
            report.maximumDiscardedNuclearWeightFraction<=1.0;
    }

    void GRC_R2GA_SpinHalfOperators(
        arma::cx_mat &sx,
        arma::cx_mat &sy,
        arma::cx_mat &sz,
        arma::cx_mat &id)
    {
        sx.zeros(2,2);
        sy.zeros(2,2);
        sz.zeros(2,2);
        id.eye(2,2);

        sx(0,1)=0.5;
        sx(1,0)=0.5;
        sy(0,1)=arma::cx_double(0.0,-0.5);
        sy(1,0)=arma::cx_double(0.0,0.5);
        sz(0,0)=0.5;
        sz(1,1)=-0.5;
    }

    bool GRC_R2GA_BuildTwoI12Comparison(
        double scale,
        RunSection::General::Resonance::ResonanceLineSet &exact,
        RunSection::General::Resonance::ResonanceLineSet &hybrid)
    {
        using namespace RunSection::General::Resonance;

        arma::cx_mat sx,sy,sz,id;
        GRC_R2GA_SpinHalfOperators(sx,sy,sz,id);

        const double electronOmega=60.0;
        const double a1=scale*0.28;
        const double a2=scale*0.19;
        const double n1=-0.021;
        const double n2=-0.034;

        const arma::cx_mat coreH=
            electronOmega*sz;
        arma::cx_mat coreDensity(2,2,arma::fill::zeros);
        coreDensity(1,1)=1.0;
        const arma::sp_cx_mat coreDHdB(sz);

        HybridNuclearResonanceNucleus first,second;
        first.hyperfineCoreNuclear=
            a1*(arma::kron(sx,sx)+
                arma::kron(sy,sy)+
                arma::kron(sz,sz));
        second.hyperfineCoreNuclear=
            a2*(arma::kron(sx,sx)+
                arma::kron(sy,sy)+
                arma::kron(sz,sz));
        first.nuclearHamiltonian=n1*sz;
        second.nuclearHamiltonian=n2*sz;
        first.nuclearDHdB=
            arma::sp_cx_mat(2,2);
        second.nuclearDHdB=
            arma::sp_cx_mat(2,2);
        first.nuclearDimension=2;
        second.nuclearDimension=2;
        first.overlapThreshold=1.0e-14;
        second.overlapThreshold=1.0e-14;

        HybridNuclearResonanceRequest multi;
        multi.nuclei={first,second};

        SpectrumRequest request;
        request.microwaveFrequencyGHz=9.5;
        request.linewidth_mT=0.10;
        request.populationThreshold=1.0e-15;
        request.minimumSlope=1.0e-15;

        HybridNuclearResonanceReport report;
        std::string error;
        if (!GRC_GenerateFirstOrder(
                    arma::sp_cx_mat(coreH),
                    coreDensity,coreDHdB,
                    sx,sy,multi,request,
                    hybrid,report,error))
            return false;

        if (hybrid.fieldJacobianQualified ||
            report.productNuclearDimension!=4 ||
            report.largestDiagonalizedNuclearDimension!=2 ||
            report.maximumDiscardedNuclearWeightFraction>
                1.0e-12)
            return false;

        const arma::cx_mat fullH =
            arma::kron(
                arma::kron(coreH,id),id) +
            arma::kron(
                arma::kron(id,n1*sz),id) +
            arma::kron(
                arma::kron(id,id),n2*sz) +
            a1*(
                arma::kron(
                    arma::kron(sx,sx),id) +
                arma::kron(
                    arma::kron(sy,sy),id) +
                arma::kron(
                    arma::kron(sz,sz),id)) +
            a2*(
                arma::kron(
                    sx,arma::kron(id,sx)) +
                arma::kron(
                    sy,arma::kron(id,sy)) +
                arma::kron(
                    sz,arma::kron(id,sz)));

        const arma::cx_mat fullDensity=
            arma::kron(
                arma::kron(
                    coreDensity,0.5*id),
                0.5*id);
        const arma::sp_cx_mat fullDHdB(
            arma::kron(
                arma::kron(sz,id),id));
        const arma::cx_mat fullMuX=
            arma::kron(
                arma::kron(sx,id),id);
        const arma::cx_mat fullMuY=
            arma::kron(
                arma::kron(sy,id),id);

        return ExactResonanceSolver::Generate(
            arma::sp_cx_mat(fullH),
            fullDensity,fullDHdB,
            fullMuX,fullMuY,
            request,exact,error);
    }

    bool GRC_TestR2GAMultiNucleusExactA2Scaling()
    {
        RunSection::General::Resonance::ResonanceLineSet
            exactStrong,hybridStrong,
            exactWeak,hybridWeak;

        if (!GRC_R2GA_BuildTwoI12Comparison(
                1.0,exactStrong,hybridStrong) ||
            !GRC_R2GA_BuildTwoI12Comparison(
                0.5,exactWeak,hybridWeak))
            return false;

        const double strongError=
            GRC_StrongLineFrequencyError(
                exactStrong,hybridStrong,4);
        const double weakError=
            GRC_StrongLineFrequencyError(
                exactWeak,hybridWeak,4);

        if (!(strongError>0.0) ||
            !(weakError>0.0))
            return false;

        const double ratio=
            strongError/weakError;
        return ratio>3.0 && ratio<5.0;
    }

    bool GRC_R2GB_LineResponseNear(
        const RunSection::General::Resonance::ResonanceLine &a,
        const RunSection::General::Resonance::ResonanceLine &b,
        double tolerance,
        bool requireCoreLabels=true)
    {
        if (requireCoreLabels &&
            (a.lower!=b.lower ||
             a.upper!=b.upper))
            return false;

        const double scale=std::max({
            1.0,
            std::abs(a.omega),std::abs(b.omega),
            std::abs(a.populationDifference),
            std::abs(b.populationDifference),
            std::abs(a.dOmegaDB),std::abs(b.dOmegaDB),
            std::abs(a.dBdOmega),std::abs(b.dBdOmega),
            std::abs(a.moment.x),std::abs(b.moment.x),
            std::abs(a.moment.y),std::abs(b.moment.y),
            std::abs(a.moment.perpendicular),
            std::abs(b.moment.perpendicular)
        });

        return
            std::abs(a.omega-b.omega)<=
                tolerance*scale &&
            std::abs(
                a.populationDifference-
                b.populationDifference)<=
                tolerance*scale &&
            std::abs(
                a.dOmegaDB-b.dOmegaDB)<=
                tolerance*scale &&
            std::abs(
                a.dBdOmega-b.dBdOmega)<=
                tolerance*scale &&
            std::abs(
                a.moment.x-b.moment.x)<=
                tolerance*scale &&
            std::abs(
                a.moment.y-b.moment.y)<=
                tolerance*scale &&
            std::abs(
                a.moment.perpendicular-
                b.moment.perpendicular)<=
                tolerance*scale;
    }

    std::vector<RunSection::General::Resonance::ResonanceLine>
    GRC_R2GB_SortedResponseLines(
        const RunSection::General::Resonance::ResonanceLineSet &set,
        bool useCoreLabels=true)
    {
        auto lines=set.lines;
        std::sort(
            lines.begin(),lines.end(),
            [=](const auto &a,const auto &b)
            {
                if (useCoreLabels &&
                    a.lower!=b.lower)
                    return a.lower<b.lower;
                if (useCoreLabels &&
                    a.upper!=b.upper)
                    return a.upper<b.upper;
                if (a.omega!=b.omega)
                    return a.omega<b.omega;
                if (a.moment.perpendicular!=
                    b.moment.perpendicular)
                    return
                        a.moment.perpendicular<
                        b.moment.perpendicular;
                if (a.dOmegaDB!=b.dOmegaDB)
                    return a.dOmegaDB<b.dOmegaDB;
                return
                    a.populationDifference<
                    b.populationDifference;
            });
        return lines;
    }

    bool GRC_R2GB_BuildZfsMultiPoint(
        const GRC_ZfsHybridModel &model,
        const RunSection::General::HS::HSOrientation &orientation,
        const std::vector<double> &scales,
        bool reverseOrder,double fieldT,
        RunSection::General::Resonance::
            HybridNuclearResonancePoint &point,
        std::string &error)
    {
        using namespace RunSection::General::Resonance;

        if (scales.empty())
        {
            error =
                "R2G-B synthetic scale list is empty";
            return false;
        }

        std::vector<HybridNuclearResonancePoint>
            one(scales.size());
        for (std::size_t k=0;
             k<scales.size();++k)
        {
            if (!GRC_BuildZfsHybridPoint(
                    model,orientation,
                    scales[k],fieldT,
                    one[k],error))
                return false;
        }

        point =
            HybridNuclearResonancePoint{};
        point.coreHamiltonian =
            one.front().coreHamiltonian;
        point.coreDensity =
            one.front().coreDensity;
        point.coreDHdB =
            one.front().coreDHdB;
        point.coreMuX =
            one.front().coreMuX;
        point.coreMuY =
            one.front().coreMuY;

        if (reverseOrder)
        {
            for (auto it=one.rbegin();
                 it!=one.rend();++it)
                point.hybrid.nuclei.push_back(
                    it->hybrid.nuclei.front());
        }
        else
        {
            for (const auto &entry:one)
                point.hybrid.nuclei.push_back(
                    entry.hybrid.nuclei.front());
        }

        return true;
    }

    RunSection::General::Resonance::
        HybridNuclearResonanceFieldResponseRequest
    GRC_R2GB_Response(double fieldT,double stepT)
    {
        using namespace RunSection::General::Resonance;
        HybridNuclearResonanceFieldResponseRequest
            response;
        response.fieldT=fieldT;
        response.fieldStepT=stepT;
        response.minimumCoreStateOverlap=0.85;
        response.minimumNuclearStateOverlap=0.85;
        response.jacobianRelativeTolerance=5.0e-4;
        response.jacobianAbsoluteTolerance=2.0e-4;
        return response;
    }

    bool GRC_TestR2GBMultiNucleusN1FieldResponseParity()
    {
        using namespace RunSection::General::Resonance;

        GRC_ZfsHybridModel model;
        const auto orientation=
            GRC_Orientation(0.27,0.76,-0.34);
        const double frequency=9.5;
        const double fieldT=
            2.0*arma::datum::pi*frequency/
            (GRC_MU_B_OVER_HBAR*model.gz);

        HybridNuclearResonancePointProvider oneProvider=
            [&](double field,
                HybridNuclearResonancePoint &point,
                std::string &error)
            {
                return GRC_BuildZfsHybridPoint(
                    model,orientation,0.53,
                    field,point,error);
            };

        HybridNuclearResonancePointProvider
            multiProvider=
            [&](double field,
                HybridNuclearResonancePoint &point,
                std::string &error)
            {
                return GRC_R2GB_BuildZfsMultiPoint(
                    model,orientation,{0.53},
                    false,field,point,error);
            };

        HybridNuclearResonanceFieldResponseRequest oneResponse;
        oneResponse.fieldT=fieldT;
        oneResponse.fieldStepT=1.0e-4;
        oneResponse.minimumCoreStateOverlap=0.85;
        oneResponse.minimumNuclearStateOverlap=0.85;
        oneResponse.jacobianRelativeTolerance=5.0e-4;
        oneResponse.jacobianAbsoluteTolerance=2.0e-4;

        const auto multiResponse=
            GRC_R2GB_Response(
                fieldT,1.0e-4);

        SpectrumRequest request;
        request.microwaveFrequencyGHz=frequency;
        request.linewidth_mT=0.10;
        request.populationThreshold=1.0e-15;
        request.minimumSlope=1.0e-15;

        ResonanceLineSet one,multi;
        HybridNuclearResonanceReport report;
        std::string error;

        if (!GRC_GenerateFirstOrderFiniteDifference(
                    oneProvider,oneResponse,
                    request,one,error) ||
            !GRC_GenerateFirstOrderFiniteDifference(
                    multiProvider,multiResponse,
                    request,multi,report,error))
            return false;

        if (!one.fieldJacobianQualified ||
            !multi.fieldJacobianQualified ||
            report.nucleusCount!=1 ||
            report.productNuclearDimension!=8 ||
            report.largestDiagonalizedNuclearDimension!=8)
            return false;

        const auto a=
            GRC_R2GB_SortedResponseLines(one);
        const auto b=
            GRC_R2GB_SortedResponseLines(multi);
        if (a.size()!=b.size() || a.empty())
            return false;

        for (std::size_t i=0;i<a.size();++i)
        {
            if (!GRC_R2GB_LineResponseNear(
                    a[i],b[i],2.0e-12))
                return false;
        }
        return true;
    }

    bool GRC_TestR2GBMultiNucleusPermutationInvariant()
    {
        using namespace RunSection::General::Resonance;

        GRC_ZfsHybridModel model;
        const auto orientation=
            GRC_Orientation(-0.29,0.72,0.38);
        const double frequency=9.5;
        const double fieldT=
            2.0*arma::datum::pi*frequency/
            (GRC_MU_B_OVER_HBAR*model.gz);

        const auto makeProvider=
            [&](bool reverse)
            {
                return HybridNuclearResonancePointProvider(
                    [&,reverse](
                        double field,
                        HybridNuclearResonancePoint &point,
                        std::string &error)
                    {
                        return GRC_R2GB_BuildZfsMultiPoint(
                            model,orientation,
                            {0.43,0.67},
                            reverse,field,
                            point,error);
                    });
            };

        const auto response=
            GRC_R2GB_Response(
                fieldT,1.0e-4);

        SpectrumRequest request;
        request.microwaveFrequencyGHz=frequency;
        request.linewidth_mT=0.10;
        request.populationThreshold=1.0e-15;
        request.minimumSlope=1.0e-15;

        ResonanceLineSet ab,ba;
        HybridNuclearResonanceReport
            reportAB,reportBA;
        std::string error;

        if (!GRC_GenerateFirstOrderFiniteDifference(
                    makeProvider(false),response,
                    request,ab,reportAB,error) ||
            !GRC_GenerateFirstOrderFiniteDifference(
                    makeProvider(true),response,
                    request,ba,reportBA,error))
            return false;

        if (!ab.fieldJacobianQualified ||
            !ba.fieldJacobianQualified ||
            reportAB.productNuclearDimension!=64 ||
            reportBA.productNuclearDimension!=64 ||
            reportAB.largestDiagonalizedNuclearDimension!=8 ||
            reportBA.largestDiagonalizedNuclearDimension!=8)
            return false;

        const auto a=
            GRC_R2GB_SortedResponseLines(ab);
        const auto b=
            GRC_R2GB_SortedResponseLines(ba);
        if (a.size()!=b.size() || a.empty())
            return false;

        for (std::size_t i=0;i<a.size();++i)
        {
            if (!GRC_R2GB_LineResponseNear(
                    a[i],b[i],5.0e-10))
                return false;
        }
        return true;
    }

    bool GRC_TestR2GBMultiNucleusStepConvergence()
    {
        using namespace RunSection::General::Resonance;

        GRC_ZfsHybridModel model;
        const auto orientation=
            GRC_Orientation(0.24,0.70,-0.45);
        const double frequency=9.5;
        const double fieldT=
            2.0*arma::datum::pi*frequency/
            (GRC_MU_B_OVER_HBAR*model.gz);

        HybridNuclearResonancePointProvider provider=
            [&](double field,
                HybridNuclearResonancePoint &point,
                std::string &error)
            {
                return GRC_R2GB_BuildZfsMultiPoint(
                    model,orientation,
                    {0.48,0.61},
                    false,field,point,error);
            };

        SpectrumRequest request;
        request.microwaveFrequencyGHz=frequency;
        request.linewidth_mT=0.10;
        request.populationThreshold=1.0e-15;
        request.minimumSlope=1.0e-15;

        ResonanceLineSet coarse,fine;
        HybridNuclearResonanceReport
            coarseReport,fineReport;
        std::string error;

        if (!GRC_GenerateFirstOrderFiniteDifference(
                    provider,
                    GRC_R2GB_Response(
                        fieldT,1.6e-4),
                    request,coarse,
                    coarseReport,error) ||
            !GRC_GenerateFirstOrderFiniteDifference(
                    provider,
                    GRC_R2GB_Response(
                        fieldT,0.8e-4),
                    request,fine,
                    fineReport,error))
            return false;

        const auto a=
            GRC_StrongestSlopeLines(
                coarse,16);
        const auto b=
            GRC_StrongestSlopeLines(
                fine,16);

        if (a.size()!=16 ||
            b.size()!=16)
            return false;

        double relative=0.0;
        for (std::size_t i=0;i<16;++i)
        {
            const double scale=
                std::max(
                    1.0,
                    std::abs(b[i].slope));
            relative=std::max(
                relative,
                std::abs(
                    a[i].slope-
                    b[i].slope)/scale);
        }

        return
            coarse.fieldJacobianQualified &&
            fine.fieldJacobianQualified &&
            relative<3.0e-4;
    }

    bool GRC_TestR2GBMultiNucleusMergingFailsClosed()
    {
        using namespace RunSection::General::Resonance;

        GRC_ZfsHybridModel model;
        const auto orientation=
            GRC_Orientation(-0.18,0.66,0.33);
        const double frequency=9.5;
        const double fieldT=
            2.0*arma::datum::pi*frequency/
            (GRC_MU_B_OVER_HBAR*model.gz);

        HybridNuclearResonancePointProvider provider=
            [&](double field,
                HybridNuclearResonancePoint &point,
                std::string &error)
            {
                if (!GRC_R2GB_BuildZfsMultiPoint(
                        model,orientation,
                        {0.45,0.58},
                        false,field,
                        point,error))
                    return false;
                point.hybrid.
                    mergeFrequencyToleranceRadNs=
                    1.0e-9;
                return true;
            };

        SpectrumRequest request;
        request.microwaveFrequencyGHz=frequency;
        request.linewidth_mT=0.10;
        request.populationThreshold=1.0e-15;
        request.minimumSlope=1.0e-15;

        ResonanceLineSet lines;
        HybridNuclearResonanceReport report;
        std::string error;

        if (GRC_GenerateFirstOrderFiniteDifference(
                    provider,
                    GRC_R2GB_Response(
                        fieldT,1.0e-4),
                    request,lines,
                    report,error))
            return false;

        return
            !lines.fieldJacobianQualified &&
            error ==
                "multi-nucleus field response requires unmerged center components";
    }

    bool GRC_R2GB_BuildConditionalI12Point(
        double fieldT,
        RunSection::General::Resonance::
            HybridNuclearResonancePoint &point)
    {
        using namespace RunSection::General::Resonance;

        arma::cx_mat sx,sy,sz,id;
        GRC_R2GA_SpinHalfOperators(
            sx,sy,sz,id);

        constexpr double gammaE=175.0;
        constexpr double gamma1=-0.071;
        constexpr double gamma2=-0.053;
        constexpr double q1=0.024;
        constexpr double q2=-0.031;
        constexpr double a1=0.32;
        constexpr double a2=-0.21;

        point=
            HybridNuclearResonancePoint{};
        point.coreHamiltonian=
            arma::sp_cx_mat(
                gammaE*fieldT*sz);
        point.coreDensity.zeros(2,2);
        point.coreDensity(1,1)=1.0;
        point.coreDHdB=
            arma::sp_cx_mat(gammaE*sz);
        point.coreMuX=sx;
        point.coreMuY=sy;

        HybridNuclearResonanceNucleus first,second;
        first.hyperfineCoreNuclear=
            a1*arma::kron(sz,sz);
        second.hyperfineCoreNuclear=
            a2*arma::kron(sz,sz);

        first.nuclearHamiltonian=
            gamma1*fieldT*sz+
            q1*sx;
        second.nuclearHamiltonian=
            gamma2*fieldT*sz+
            q2*sx;

        first.nuclearDHdB=
            arma::sp_cx_mat(
                gamma1*sz);
        second.nuclearDHdB=
            arma::sp_cx_mat(
                gamma2*sz);

        first.nuclearDimension=2;
        second.nuclearDimension=2;
        first.overlapThreshold=1.0e-14;
        second.overlapThreshold=1.0e-14;

        point.hybrid.nuclei={
            first,second};
        return true;
    }

    bool GRC_R2GB_BuildConditionalI12ExactLines(
        double fieldT,
        const RunSection::General::Resonance::
            SpectrumRequest &request,
        RunSection::General::Resonance::
            ResonanceLineSet &lines)
    {
        using namespace RunSection::General::Resonance;

        arma::cx_mat sx,sy,sz,id;
        GRC_R2GA_SpinHalfOperators(
            sx,sy,sz,id);

        constexpr double gammaE=175.0;
        constexpr double gamma1=-0.071;
        constexpr double gamma2=-0.053;
        constexpr double q1=0.024;
        constexpr double q2=-0.031;
        constexpr double a1=0.32;
        constexpr double a2=-0.21;

        const arma::cx_mat coreH=
            gammaE*fieldT*sz;
        const arma::cx_mat n1=
            gamma1*fieldT*sz+
            q1*sx;
        const arma::cx_mat n2=
            gamma2*fieldT*sz+
            q2*sx;

        const arma::cx_mat fullH=
            arma::kron(
                arma::kron(coreH,id),id)+
            arma::kron(
                arma::kron(id,n1),id)+
            arma::kron(
                arma::kron(id,id),n2)+
            a1*arma::kron(
                arma::kron(sz,sz),id)+
            a2*arma::kron(
                sz,arma::kron(id,sz));

        arma::cx_mat coreDensity(
            2,2,arma::fill::zeros);
        coreDensity(1,1)=1.0;
        const arma::cx_mat fullDensity=
            arma::kron(
                arma::kron(
                    coreDensity,0.5*id),
                0.5*id);

        const arma::cx_mat derivative=
            gammaE*arma::kron(
                arma::kron(sz,id),id)+
            gamma1*arma::kron(
                arma::kron(id,sz),id)+
            gamma2*arma::kron(
                id,arma::kron(id,sz));

        const arma::cx_mat muX=
            arma::kron(
                arma::kron(sx,id),id);
        const arma::cx_mat muY=
            arma::kron(
                arma::kron(sy,id),id);

        std::string error;
        return ExactResonanceSolver::Generate(
            arma::sp_cx_mat(fullH),
            fullDensity,
            arma::sp_cx_mat(derivative),
            muX,muY,
            request,lines,error);
    }

    bool GRC_TestR2GBConditionalTwoI12ExactFieldResponseParity()
    {
        using namespace RunSection::General::Resonance;

        const double fieldT=0.34;

        SpectrumRequest request;
        request.microwaveFrequencyGHz=9.5;
        request.linewidth_mT=0.10;
        request.populationThreshold=1.0e-14;
        request.minimumSlope=1.0e-12;

        ResonanceLineSet exact,hybrid;
        if (!GRC_R2GB_BuildConditionalI12ExactLines(
                fieldT,request,exact))
            return false;

        HybridNuclearResonancePointProvider provider=
            [](double field,
               HybridNuclearResonancePoint &point,
               std::string &error)
            {
                error.clear();
                return
                    GRC_R2GB_BuildConditionalI12Point(
                        field,point);
            };

        auto response=
            GRC_R2GB_Response(
                fieldT,1.0e-4);
        response.minimumCoreStateOverlap=0.99;
        response.minimumNuclearStateOverlap=0.95;
        response.jacobianRelativeTolerance=1.0e-7;
        response.jacobianAbsoluteTolerance=1.0e-7;

        HybridNuclearResonanceReport report;
        std::string error;
        if (!GRC_GenerateFirstOrderFiniteDifference(
                    provider,response,
                    request,hybrid,
                    report,error))
            return false;

        if (!exact.fieldJacobianQualified ||
            !hybrid.fieldJacobianQualified ||
            report.productNuclearDimension!=4 ||
            report.largestDiagonalizedNuclearDimension!=2 ||
            report.maximumDiscardedNuclearWeightFraction>
                1.0e-12)
            return false;

        auto a=
            GRC_R2GB_SortedResponseLines(
                exact,false);
        auto b=
            GRC_R2GB_SortedResponseLines(
                hybrid,false);

        a.erase(
            std::remove_if(
                a.begin(),a.end(),
                [](const auto &line)
                {
                    return
                        GRC_LineWeight(line)<=
                        1.0e-14;
                }),
            a.end());
        b.erase(
            std::remove_if(
                b.begin(),b.end(),
                [](const auto &line)
                {
                    return
                        GRC_LineWeight(line)<=
                        1.0e-14;
                }),
            b.end());

        if (a.size()!=b.size() ||
            a.empty())
            return false;

        for (std::size_t i=0;i<a.size();++i)
        {
            if (!GRC_R2GB_LineResponseNear(
                    a[i],b[i],
                    2.0e-6,false))
                return false;
        }

        return true;
    }

    bool GRC_R2INear(
        double a,double b,double tolerance)
    {
        const double scale=std::max({
            1.0,std::abs(a),std::abs(b)});
        return std::abs(a-b)<=tolerance*scale;
    }

    bool GRC_TestR2IBuilderResolvedChannelSum()
    {
        using namespace RunSection::General::Resonance;

        auto e1=std::make_shared<SpinAPI::Spin>(
            "E1",
            "type=electron;spin=1/2;"
            "tensor=anisotropic(1.91 2.07 2.31);");
        auto e2=std::make_shared<SpinAPI::Spin>(
            "E2",
            "type=electron;spin=1/2;"
            "tensor=anisotropic(1.97 2.04 2.22);");

        auto b1=std::make_shared<SpinAPI::Interaction>(
            "B1",
            "type=zeeman;spins=E1;field=0 0 0.34;"
            "orientation=0.27,0.58,-0.31;"
            "ignoretensors=false;"
            "commonprefactor=true;prefactor=1.0;");
        auto b2=std::make_shared<SpinAPI::Interaction>(
            "B2",
            "type=zeeman;spins=E2;field=0 0 0.34;"
            "orientation=-0.19,0.46,0.38;"
            "ignoretensors=false;"
            "commonprefactor=true;prefactor=1.0;");

        auto system=std::make_shared<SpinAPI::SpinSystem>(
            "R2IBuilder");
        system->Add(e1); system->Add(e2);
        system->Add(b1); system->Add(b2);
        if (!system->ValidateInteractions().empty())
            return false;

        SpinAPI::SpinSpace space(
            std::vector<SpinAPI::spin_ptr>{e1,e2});
        space.UseSuperoperatorSpace(false);
        space.UseFullTensorRotation(true);

        const auto orientation=
            GRC_Orientation(0.41,0.73,-0.22);

        std::vector<ResonanceDetectionOperator> channels;
        arma::cx_mat totalX,totalY;
        std::string error;

        if (!ResonanceMagneticMomentBuilder::
                BuildTransverseChannels(
                    space,{{e1,b1},{e2,b2}},
                    orientation.frameToLab,true,
                    channels,error) ||
            !ResonanceMagneticMomentBuilder::
                BuildTransverse(
                    space,{{e1,b1},{e2,b2}},
                    orientation.frameToLab,true,
                    totalX,totalY,error) ||
            channels.size()!=2)
            return false;

        arma::cx_mat sumX(
            totalX.n_rows,totalX.n_cols,
            arma::fill::zeros);
        arma::cx_mat sumY(
            totalY.n_rows,totalY.n_cols,
            arma::fill::zeros);
        for (const auto &channel:channels)
        {
            sumX+=channel.x;
            sumY+=channel.y;
        }

        arma::cx_mat oneX,oneY;
        if (!ResonanceMagneticMomentBuilder::
                BuildTransverse(
                    space,{{e1,b1}},
                    orientation.frameToLab,true,
                    oneX,oneY,error))
            return false;

        const double scale=std::max({
            1.0,arma::norm(totalX,"fro"),
            arma::norm(totalY,"fro")
        });

        return
            arma::norm(sumX-totalX,"fro")<=
                1.0e-13*scale &&
            arma::norm(sumY-totalY,"fro")<=
                1.0e-13*scale &&
            arma::norm(
                channels[0].x-oneX,"fro")<=
                1.0e-13*scale &&
            arma::norm(
                channels[0].y-oneY,"fro")<=
                1.0e-13*scale;
    }

    bool GRC_TestR2ITransitionMomentDecomposition()
    {
        using namespace RunSection::General::Resonance;

        ResonanceDetectionOperator a,b;
        a.x.zeros(2,2); a.y.zeros(2,2);
        b.x.zeros(2,2); b.y.zeros(2,2);

        a.x(0,1)=arma::cx_double(1.0,0.30);
        a.y(0,1)=arma::cx_double(0.20,-0.40);
        b.x(0,1)=arma::cx_double(-0.35,0.15);
        b.y(0,1)=arma::cx_double(0.55,0.25);

        const arma::cx_mat totalX=a.x+b.x;
        const arma::cx_mat totalY=a.y+b.y;

        TransitionMoment moment;
        std::string error;
        if (!ResonanceTransitionMoments::EvaluateResolved(
                totalX,totalY,{a,b},0,1,
                moment,error) ||
            moment.channels.size()!=2)
            return false;

        const arma::cx_double I(0.0,1.0);
        const auto ax=a.x(0,1);
        const auto ay=a.y(0,1);
        const auto bx=b.x(0,1);
        const auto by=b.y(0,1);

        const double totalIX=std::norm(ax+bx);
        const double totalIY=std::norm(ay+by);
        const double sumIX=std::norm(ax)+std::norm(bx);
        const double sumIY=std::norm(ay)+std::norm(by);

        return
            ResonanceTransitionMoments::IsFinite(moment) &&
            GRC_R2INear(moment.x,totalIX,1.0e-14) &&
            GRC_R2INear(moment.y,totalIY,1.0e-14) &&
            GRC_R2INear(
                moment.crossX,
                totalIX-sumIX,1.0e-14) &&
            GRC_R2INear(
                moment.crossY,
                totalIY-sumIY,1.0e-14) &&
            GRC_R2INear(
                moment.channels[0].plus,
                std::norm(ax+I*ay),1.0e-14) &&
            GRC_R2INear(
                moment.channels[0].minus,
                std::norm(ax-I*ay),1.0e-14) &&
            GRC_R2INear(
                moment.channels[1].plus,
                std::norm(bx+I*by),1.0e-14) &&
            GRC_R2INear(
                moment.channels[1].minus,
                std::norm(bx-I*by),1.0e-14);
    }

    bool GRC_TestR2IExactResolvedLineParity()
    {
        using namespace RunSection::General::Resonance;

        arma::cx_mat H(2,2,arma::fill::zeros);
        H(1,1)=60.0;
        arma::cx_mat rho(2,2,arma::fill::zeros);
        rho(0,0)=1.0;
        arma::cx_mat derivative(2,2,arma::fill::zeros);
        derivative(1,1)=175.0;

        ResonanceDetectionOperator a,b;
        a.x.zeros(2,2); a.y.zeros(2,2);
        b.x.zeros(2,2); b.y.zeros(2,2);
        a.x(0,1)=arma::cx_double(0.7,0.2);
        a.x(1,0)=std::conj(a.x(0,1));
        a.y(0,1)=arma::cx_double(0.1,-0.3);
        a.y(1,0)=std::conj(a.y(0,1));
        b.x(0,1)=arma::cx_double(-0.2,0.4);
        b.x(1,0)=std::conj(b.x(0,1));
        b.y(0,1)=arma::cx_double(0.5,0.15);
        b.y(1,0)=std::conj(b.y(0,1));

        const arma::cx_mat totalX=a.x+b.x;
        const arma::cx_mat totalY=a.y+b.y;

        SpectrumRequest request;
        request.microwaveFrequencyGHz=9.5;
        request.linewidth_mT=0.1;
        request.populationThreshold=1.0e-15;
        request.minimumSlope=1.0e-15;

        ResonanceLineSet frozen,resolved;
        std::string error;
        if (!ExactResonanceSolver::Generate(
                arma::sp_cx_mat(H),rho,
                arma::sp_cx_mat(derivative),
                totalX,totalY,request,
                frozen,error) ||
            !ExactResonanceSolver::Generate(
                arma::sp_cx_mat(H),rho,
                arma::sp_cx_mat(derivative),
                totalX,totalY,request,
                resolved,error,{a,b}) ||
            frozen.lines.size()!=1 ||
            resolved.lines.size()!=1)
            return false;

        const auto &x=frozen.lines.front();
        const auto &y=resolved.lines.front();

        return
            y.moment.channels.size()==2 &&
            x.lower==y.lower &&
            x.upper==y.upper &&
            GRC_R2INear(x.omega,y.omega,1.0e-14) &&
            GRC_R2INear(
                x.populationDifference,
                y.populationDifference,1.0e-14) &&
            GRC_R2INear(
                x.dOmegaDB,y.dOmegaDB,1.0e-14) &&
            GRC_R2INear(
                x.dBdOmega,y.dBdOmega,1.0e-14) &&
            GRC_R2INear(
                x.moment.x,y.moment.x,1.0e-14) &&
            GRC_R2INear(
                x.moment.y,y.moment.y,1.0e-14) &&
            GRC_R2INear(
                x.moment.perpendicular,
                y.moment.perpendicular,1.0e-14);
    }

    bool GRC_R2IBuildSpectatorPoint(
        std::size_t nucleusCount,
        RunSection::General::Resonance::
            HybridNuclearResonancePoint &point)
    {
        using namespace RunSection::General::Resonance;

        if (nucleusCount==0)
            return false;

        arma::cx_mat sx,sy,sz,id;
        GRC_R2GA_SpinHalfOperators(
            sx,sy,sz,id);

        point=HybridNuclearResonancePoint{};
        point.coreHamiltonian=
            arma::sp_cx_mat(60.0*sz);
        point.coreDensity.zeros(2,2);
        point.coreDensity(1,1)=1.0;
        point.coreDHdB=
            arma::sp_cx_mat(175.0*sz);

        ResonanceDetectionOperator a,b;
        a.x=0.65*sx;
        a.y=0.25*sy;
        b.x=0.35*sx;
        b.y=0.75*sy;
        point.coreDetectionChannels={a,b};
        point.coreMuX=a.x+b.x;
        point.coreMuY=a.y+b.y;

        for (std::size_t k=0;k<nucleusCount;++k)
        {
            HybridNuclearResonanceNucleus nucleus;
            nucleus.hyperfineCoreNuclear.zeros(4,4);
            nucleus.nuclearHamiltonian.zeros(2,2);
            nucleus.nuclearDHdB=
                arma::sp_cx_mat(2,2);
            nucleus.nuclearDimension=2;
            nucleus.overlapThreshold=1.0e-14;
            point.hybrid.nuclei.push_back(nucleus);
        }
        return true;
    }

    bool GRC_TestR2IHybridResolvedScaling()
    {
        using namespace RunSection::General::Resonance;

        HybridNuclearResonancePoint point;
        if (!GRC_R2IBuildSpectatorPoint(1,point))
            return false;

        point.hybrid.nuclei.front().
            fieldIndependentProjection=true;

        SpectrumRequest request;
        request.microwaveFrequencyGHz=9.5;
        request.linewidth_mT=0.1;
        request.populationThreshold=1.0e-15;
        request.minimumSlope=1.0e-15;

        ResonanceLineSet lines;
        HybridNuclearResonanceReport report;
        std::string error;
        if (!HybridNuclearResonanceSolver::
                GenerateFirstOrder(
                    point,request,lines,
                    report,error) ||
            !lines.fieldJacobianQualified ||
            lines.lines.empty())
            return false;

        for (const auto &line:lines.lines)
        {
            if (line.moment.channels.size()!=2 ||
                !ResonanceTransitionMoments::
                    IsFinite(line.moment))
                return false;

            const double sumX=
                line.moment.channels[0].x+
                line.moment.channels[1].x;
            const double sumY=
                line.moment.channels[0].y+
                line.moment.channels[1].y;

            if (!GRC_R2INear(
                    line.moment.crossX,
                    line.moment.x-sumX,
                    1.0e-13) ||
                !GRC_R2INear(
                    line.moment.crossY,
                    line.moment.y-sumY,
                    1.0e-13))
                return false;
        }
        return true;
    }

    bool GRC_TestR2IMergedSpectatorChannelConservation()
    {
        using namespace RunSection::General::Resonance;

        HybridNuclearResonancePoint one,two;
        if (!GRC_R2IBuildSpectatorPoint(1,one) ||
            !GRC_R2IBuildSpectatorPoint(2,two))
            return false;

        one.hybrid.mergeFrequencyToleranceRadNs=1.0e-12;
        two.hybrid.mergeFrequencyToleranceRadNs=1.0e-12;

        SpectrumRequest request;
        request.microwaveFrequencyGHz=9.5;
        request.linewidth_mT=0.1;
        request.populationThreshold=1.0e-15;

        ResonanceLineSet a,b;
        HybridNuclearResonanceReport ra,rb;
        std::string error;
        if (!HybridNuclearResonanceSolver::
                GenerateFirstOrder(
                    one,request,a,ra,error) ||
            !HybridNuclearResonanceSolver::
                GenerateFirstOrder(
                    two,request,b,rb,error) ||
            !rb.mergingApplied)
            return false;

        const auto x=
            GRC_R2GA_SortedFixedFieldLines(a);
        const auto y=
            GRC_R2GA_SortedFixedFieldLines(b);
        if (x.size()!=y.size() || x.empty())
            return false;

        for (std::size_t i=0;i<x.size();++i)
        {
            if (x[i].moment.channels.size()!=2 ||
                y[i].moment.channels.size()!=2)
                return false;

            const double scale=std::max({
                1.0,
                std::abs(
                    x[i].populationDifference*
                    x[i].moment.x),
                std::abs(
                    y[i].populationDifference*
                    y[i].moment.x)
            });

            const auto weightedNear=
                [&](double xv,double yv)
                {
                    return std::abs(
                        x[i].populationDifference*xv-
                        y[i].populationDifference*yv)
                        <=2.0e-12*scale;
                };

            if (!weightedNear(
                    x[i].moment.x,
                    y[i].moment.x) ||
                !weightedNear(
                    x[i].moment.y,
                    y[i].moment.y) ||
                !weightedNear(
                    x[i].moment.crossX,
                    y[i].moment.crossX) ||
                !weightedNear(
                    x[i].moment.crossY,
                    y[i].moment.crossY))
                return false;

            for (std::size_t k=0;k<2;++k)
            {
                if (!weightedNear(
                        x[i].moment.channels[k].x,
                        y[i].moment.channels[k].x) ||
                    !weightedNear(
                        x[i].moment.channels[k].y,
                        y[i].moment.channels[k].y) ||
                    !weightedNear(
                        x[i].moment.channels[k].plus,
                        y[i].moment.channels[k].plus) ||
                    !weightedNear(
                        x[i].moment.channels[k].minus,
                        y[i].moment.channels[k].minus))
                    return false;
            }
        }
        return true;
    }

    bool GRC_LineSetsNear(
        const RunSection::General::Resonance::ResonanceLineSet &a,
        const RunSection::General::Resonance::ResonanceLineSet &b,
        double tolerance);

    bool GRC_R2HCanonicalNucleusNear(
        const RunSection::General::Resonance::
            HybridNuclearResonanceNucleus &a,
        const RunSection::General::Resonance::
            HybridNuclearResonanceNucleus &b,
        double tolerance)
    {
        const auto denseNear=
            [&](const arma::cx_mat &x,
                const arma::cx_mat &y)
            {
                if (x.n_rows!=y.n_rows ||
                    x.n_cols!=y.n_cols)
                    return false;
                const double scale=std::max({
                    1.0,
                    arma::norm(x,"fro"),
                    arma::norm(y,"fro")
                });
                return
                    arma::norm(x-y,"fro")<=
                    tolerance*scale;
            };

        const auto sparseNear=
            [&](const arma::sp_cx_mat &x,
                const arma::sp_cx_mat &y)
            {
                return denseNear(
                    arma::cx_mat(x),
                    arma::cx_mat(y));
            };

        return
            denseNear(
                a.hyperfineCoreNuclear,
                b.hyperfineCoreNuclear) &&
            denseNear(
                a.nuclearHamiltonian,
                b.nuclearHamiltonian) &&
            sparseNear(
                a.nuclearDHdB,
                b.nuclearDHdB) &&
            a.nuclearDimension==
                b.nuclearDimension &&
            std::abs(
                a.overlapThreshold-
                b.overlapThreshold)<=
                tolerance &&
            a.fieldIndependentProjection==
                b.fieldIndependentProjection;
    }

    bool GRC_R2HCanonicalPointNear(
        const RunSection::General::Resonance::
            HybridNuclearResonancePoint &a,
        const RunSection::General::Resonance::
            HybridNuclearResonancePoint &b,
        double tolerance)
    {
        const auto sparseNear=
            [&](const arma::sp_cx_mat &x,
                const arma::sp_cx_mat &y)
            {
                if (x.n_rows!=y.n_rows ||
                    x.n_cols!=y.n_cols)
                    return false;
                const arma::cx_mat xd(x),yd(y);
                const double scale=std::max({
                    1.0,
                    arma::norm(xd,"fro"),
                    arma::norm(yd,"fro")
                });
                return arma::norm(
                    xd-yd,"fro")<=
                    tolerance*scale;
            };

        const auto denseNear=
            [&](const arma::cx_mat &x,
                const arma::cx_mat &y)
            {
                if (x.n_rows!=y.n_rows ||
                    x.n_cols!=y.n_cols)
                    return false;
                const double scale=std::max({
                    1.0,
                    arma::norm(x,"fro"),
                    arma::norm(y,"fro")
                });
                return arma::norm(
                    x-y,"fro")<=
                    tolerance*scale;
            };

        if (!sparseNear(
                a.coreHamiltonian,
                b.coreHamiltonian) ||
            !denseNear(
                a.coreDensity,
                b.coreDensity) ||
            !sparseNear(
                a.coreDHdB,
                b.coreDHdB) ||
            !denseNear(
                a.coreMuX,b.coreMuX) ||
            !denseNear(
                a.coreMuY,b.coreMuY) ||
            a.hybrid.nuclei.size()!=
                b.hybrid.nuclei.size() ||
            a.hybrid.maximumComponentsPerCoreTransition !=
                b.hybrid.maximumComponentsPerCoreTransition ||
            std::abs(
                a.hybrid.minimumCumulativeOverlapWeight-
                b.hybrid.minimumCumulativeOverlapWeight)>
                tolerance ||
            std::abs(
                a.hybrid.mergeFrequencyToleranceRadNs-
                b.hybrid.mergeFrequencyToleranceRadNs)>
                tolerance)
            return false;

        for (std::size_t k=0;
             k<a.hybrid.nuclei.size();++k)
        {
            if (!GRC_R2HCanonicalNucleusNear(
                    a.hybrid.nuclei[k],
                    b.hybrid.nuclei[k],
                    tolerance))
                return false;
        }

        return true;
    }

    std::shared_ptr<SpinAPI::SpinSystem>
    GRC_R2HBuildPhysicalSystem(
        double fieldT,
        const std::vector<double> &couplings)
    {
        auto system=
            std::make_shared<SpinAPI::SpinSystem>(
                "R2HCanonicalPhysical");

        auto electron=
            std::make_shared<SpinAPI::Spin>(
                "E",
                "type=electron;spin=1/2;"
                "tensor=isotropic(2.0023);");
        system->Add(electron);

        std::ostringstream bprops;
        bprops << std::setprecision(17)
               << "type=zeeman;spins=E;"
               << "field=0 0 " << fieldT << ";"
               << "ignoretensors=false;"
               << "commonprefactor=true;"
               << "prefactor=1.0;";
        auto b0=
            std::make_shared<SpinAPI::Interaction>(
                "B0",bprops.str());
        system->Add(b0);

        for (std::size_t k=0;
             k<couplings.size();++k)
        {
            const std::string suffix=
                std::to_string(k+1);
            const std::string name=
                "V"+suffix;

            auto nucleus=
                std::make_shared<SpinAPI::Spin>(
                    name,
                    "type=nucleus;spin=7/2;"
                    "isotope=51V;"
                    "tensor=isotropic(1.0);");
            system->Add(nucleus);

            std::ostringstream aprops;
            aprops << std::setprecision(17)
                   << "type=hyperfine;"
                   << "group1=E;group2=" << name << ";"
                   << "tensor=isotropic("
                   << couplings[k] << ");"
                   << "commonprefactor=false;"
                   << "prefactor=1.0;";
            auto hfc=
                std::make_shared<SpinAPI::Interaction>(
                    "A"+suffix,
                    aprops.str());
            system->Add(hfc);

            arma::vec field(
                3,arma::fill::zeros);
            field(2)=fieldT;

            SpinAPI::interaction_ptr nz;
            std::string error;
            if (!SpinAPI::NuclearZeeman::
                    CreateInteraction(
                        "NZ"+suffix,
                        nucleus,field,
                        nz,error))
                return nullptr;
            system->Add(nz);
        }

        auto up=
            std::make_shared<SpinAPI::State>(
                "Up","spin(E)=|1/2>;");
        system->Add(up);

        if (!system->ValidateInteractions().empty() ||
            !up->ParseFromSystem(*system))
            return nullptr;

        return system;
    }

    bool GRC_R2HPreparePhysical(
        double fieldT,
        const std::vector<double> &couplings,
        bool reverseOrder,
        RunSection::General::Resonance::
            HybridNuclearResonancePoint &point,
        std::string &error,
        RunSection::General::Resonance::
            HybridNuclearResonancePartition
                *partitionOut=nullptr)
    {
        using namespace RunSection::General::Resonance;

        auto system=
            GRC_R2HBuildPhysicalSystem(
                fieldT,couplings);
        if (system==nullptr)
        {
            error =
                "failed to build canonical R2H physical system";
            return false;
        }

        auto electron=
            system->spins_find("E");
        auto b0=
            system->interactions_find("B0");
        auto up=
            system->states_find("Up");

        HybridNuclearResonancePartition partition;
        partition.system=system;
        partition.exactCoreSpins={electron};
        partition.exactCoreInteractions={b0};
        partition.exactCoreFieldInteractions={b0};
        partition.detectionTerms={{electron,b0}};
        partition.exactCoreState=up;

        for (std::size_t k=0;
             k<couplings.size();++k)
        {
            const std::size_t index=
                reverseOrder
                ? couplings.size()-1-k
                : k;
            const std::string suffix=
                std::to_string(index+1);

            HybridNuclearResonanceNucleusPartition factor;
            factor.nucleus=
                system->spins_find(
                    "V"+suffix);
            factor.hyperfine=
                system->interactions_find(
                    "A"+suffix);
            auto nz=
                system->interactions_find(
                    "NZ"+suffix);
            factor.nuclearInteractions={nz};
            factor.nuclearFieldInteractions={nz};
            factor.overlapThreshold=1.0e-14;
            partition.nuclei.push_back(
                factor);
        }

        if (partitionOut!=nullptr)
            *partitionOut=partition;

        return HybridNuclearResonancePreparation::
            BuildPoint(
                partition,
                GRC_IdentityOrientation(),
                fieldT,point,error);
    }

    bool GRC_TestR2HCanonicalCardinalityContract()
    {
        using namespace RunSection::General::Resonance;

        HybridNuclearResonancePoint one,two;
        std::string error;

        if (!GRC_R2HPreparePhysical(
                0.34,{0.0047},false,
                one,error) ||
            !GRC_R2HPreparePhysical(
                0.34,{0.0047,0.0068},false,
                two,error))
            return false;

        return
            one.hybrid.nuclei.size()==1 &&
            two.hybrid.nuclei.size()==2 &&
            one.hybrid.nuclei.front().
                nuclearDimension==8 &&
            two.hybrid.nuclei[0].
                nuclearDimension==8 &&
            two.hybrid.nuclei[1].
                nuclearDimension==8;
    }

    bool GRC_TestR2HCanonicalTwoV51PointParity()
    {
        using namespace RunSection::General::Resonance;

        const double fieldT=0.34;
        const double a1=0.0047;
        const double a2=0.0068;

        HybridNuclearResonancePoint
            prepared,manual,p1,p2;
        std::string error;

        if (!GRC_R2HPreparePhysical(
                fieldT,{a1,a2},false,
                prepared,error) ||
            !GRC_R2HPreparePhysical(
                fieldT,{a1},false,
                p1,error) ||
            !GRC_R2HPreparePhysical(
                fieldT,{a2},false,
                p2,error))
            return false;

        manual=p1;
        manual.hybrid.nuclei={
            p1.hybrid.nuclei.front(),
            p2.hybrid.nuclei.front()};

        return GRC_R2HCanonicalPointNear(
            prepared,manual,1.0e-13);
    }

    bool GRC_TestR2HCanonicalTwoV51FieldResponseParity()
    {
        using namespace RunSection::General::Resonance;

        const double fieldT=0.34;
        const double a1=0.0049;
        const double a2=0.0061;

        HybridNuclearResonancePointProvider preparedProvider=
            [&](double field,
                HybridNuclearResonancePoint &point,
                std::string &error)
            {
                return GRC_R2HPreparePhysical(
                    field,{a1,a2},false,
                    point,error);
            };

        HybridNuclearResonancePointProvider manualProvider=
            [&](double field,
                HybridNuclearResonancePoint &point,
                std::string &error)
            {
                HybridNuclearResonancePoint p1,p2;
                if (!GRC_R2HPreparePhysical(
                        field,{a1},false,
                        p1,error) ||
                    !GRC_R2HPreparePhysical(
                        field,{a2},false,
                        p2,error))
                    return false;

                point=p1;
                point.hybrid.nuclei={
                    p1.hybrid.nuclei.front(),
                    p2.hybrid.nuclei.front()};
                return true;
            };

        HybridNuclearResonanceFieldResponseRequest response;
        response.fieldT=fieldT;
        response.fieldStepT=1.0e-4;
        response.minimumCoreStateOverlap=0.99;
        response.minimumNuclearStateOverlap=0.99;
        response.jacobianRelativeTolerance=1.0e-6;
        response.jacobianAbsoluteTolerance=1.0e-6;

        SpectrumRequest request;
        request.microwaveFrequencyGHz=9.5;
        request.linewidth_mT=0.10;
        request.populationThreshold=1.0e-15;
        request.minimumSlope=1.0e-15;

        ResonanceLineSet prepared,manual;
        HybridNuclearResonanceReport
            preparedReport,manualReport;
        std::string error;

        if (!HybridNuclearResonanceSolver::
                GenerateFirstOrderFiniteDifference(
                    preparedProvider,response,
                    request,prepared,
                    preparedReport,error) ||
            !HybridNuclearResonanceSolver::
                GenerateFirstOrderFiniteDifference(
                    manualProvider,response,
                    request,manual,
                    manualReport,error))
            return false;

        return
            prepared.fieldJacobianQualified &&
            manual.fieldJacobianQualified &&
            preparedReport.nucleusCount==2 &&
            preparedReport.productNuclearDimension==64 &&
            preparedReport.
                largestDiagonalizedNuclearDimension==8 &&
            GRC_LineSetsNear(
                prepared,manual,1.0e-12);
    }

    bool GRC_TestR2HCanonicalOwnershipFailsClosed()
    {
        using namespace RunSection::General::Resonance;

        const double fieldT=0.34;
        HybridNuclearResonancePoint point;
        HybridNuclearResonancePartition valid;
        std::string error;

        if (!GRC_R2HPreparePhysical(
                fieldT,{0.0045,0.0063},
                false,point,error,&valid))
            return false;

        auto duplicate=valid;
        duplicate.nuclei.push_back(
            duplicate.nuclei.front());
        if (HybridNuclearResonancePreparation::
                BuildPoint(
                    duplicate,
                    GRC_IdentityOrientation(),
                    fieldT,point,error) ||
            error !=
                "hybrid partition perturbative nuclei must be non-empty and unique")
            return false;

        auto wrongHyperfine=valid;
        wrongHyperfine.nuclei[1].hyperfine=
            wrongHyperfine.nuclei[0].hyperfine;
        if (HybridNuclearResonancePreparation::
                BuildPoint(
                    wrongHyperfine,
                    GRC_IdentityOrientation(),
                    fieldT,point,error) ||
            error !=
                "hybrid perturbative hyperfine interaction does not connect nucleus and exact core exclusively")
            return false;

        auto unowned=valid;
        unowned.nuclei[1].
            nuclearInteractions.clear();
        unowned.nuclei[1].
            nuclearFieldInteractions.clear();
        if (HybridNuclearResonancePreparation::
                BuildPoint(
                    unowned,
                    GRC_IdentityOrientation(),
                    fieldT,point,error) ||
            error !=
                "hybrid partition must own every SpinSystem interaction exactly once")
            return false;

        auto polarized=
            std::make_shared<SpinAPI::State>(
                "PolarizedV2",
                "spin(E)=|1/2>;"
                "spin(V2)=|7/2>;");
        valid.system->Add(polarized);
        if (!polarized->ParseFromSystem(
                *valid.system))
            return false;

        auto polarizedPartition=valid;
        polarizedPartition.exactCoreState=
            polarized;
        if (HybridNuclearResonancePreparation::
                BuildPoint(
                    polarizedPartition,
                    GRC_IdentityOrientation(),
                    fieldT,point,error) ||
            error !=
                "hybrid perturbative nucleus reference state must be unpolarized and unspecified")
            return false;

        auto invalidControl=valid;
        invalidControl.
            minimumCumulativeOverlapWeight=1.1;
        if (HybridNuclearResonancePreparation::
                BuildPoint(
                    invalidControl,
                    GRC_IdentityOrientation(),
                    fieldT,point,error) ||
            error !=
                "hybrid composition controls are invalid")
            return false;

        return true;
    }

    bool GRC_TestNuclearZeemanV51Constants()
    {
        SpinAPI::NuclearIsotopeData a,b,c;
        std::string error;

        if (!SpinAPI::NuclearIsotopeRegistry::Lookup(
                "51V",a,error) ||
            !SpinAPI::NuclearIsotopeRegistry::Lookup(
                "V-51",b,error) ||
            !SpinAPI::NuclearIsotopeRegistry::Lookup(
                "^51v",c,error))
            return false;

        const double expectedG=
            1.4710587714285714;
        const double expectedGamma=
            0.07045513257526152;

        return
            a.canonicalName=="51V" &&
            a.elementSymbol=="V" &&
            a.massNumber==51 &&
            a.twoI==7 &&
            std::abs(
                a.magneticMomentNuclearMagnetons-
                5.1487057)<1.0e-13 &&
            b.canonicalName==a.canonicalName &&
            c.canonicalName==a.canonicalName &&
            std::abs(
                SpinAPI::NuclearZeeman::NuclearG(a)-
                expectedG)<1.0e-13 &&
            std::abs(
                SpinAPI::NuclearZeeman::
                    GyromagneticMagnitudeRadNsPerT(a)-
                expectedGamma)<1.0e-13 &&
            std::abs(
                SpinAPI::NuclearZeeman::
                    HamiltonianPrefactorRadNsPerT(a)+
                expectedGamma)<1.0e-13;
    }

    bool GRC_TestNuclearZeemanSignedHamiltonian()
    {
        auto nucleus=
            std::make_shared<SpinAPI::Spin>(
                "V",
                "type=nucleus;spin=7/2;"
                "isotope=51V;"
                "tensor=isotropic(1.0);");

        arma::vec field(3,arma::fill::zeros);
        field(2)=1.0;

        SpinAPI::interaction_ptr interaction;
        std::string error;
        if (!SpinAPI::NuclearZeeman::CreateInteraction(
                "NZ",nucleus,field,
                interaction,error))
            return false;

        // NuclearZeeman::CreateInteraction follows the normal SpinAPI
        // lifecycle and returns an unbound Interaction.  Bind it exactly once
        // through SpinSystem::ValidateInteractions before operator use.
        if (!interaction->Group1().empty() ||
            !interaction->Group2().empty())
            return false;

        auto system=
            std::make_shared<SpinAPI::SpinSystem>(
                "NuclearZeemanSigned");
        system->Add(nucleus);
        system->Add(interaction);
        if (!system->ValidateInteractions().empty() ||
            !SpinAPI::NuclearZeeman::ValidateInteraction(
                nucleus,interaction,error))
            return false;

        if (interaction->AddCommonPrefactor() ||
            !interaction->IgnoreTensors() ||
            !(interaction->Prefactor()<0.0))
            return false;

        SpinAPI::SpinSpace space(nucleus);
        space.UseSuperoperatorSpace(false);

        arma::cx_mat h;
        if (!space.InteractionOperator(
                interaction,h))
            return false;

        const arma::cx_mat iz=
            arma::conv_to<arma::cx_mat>::from(
                nucleus->Sz());
        const arma::cx_mat expected=
            interaction->Prefactor()*iz;
        const double scale=std::max(
            1.0,arma::norm(expected,"fro"));

        return
            arma::norm(h-expected,"fro") <=
                1.0e-13*scale;
    }

    bool GRC_TestNuclearZeemanFailsClosed()
    {
        std::string error;
        SpinAPI::interaction_ptr interaction;

        auto noIsotope=
            std::make_shared<SpinAPI::Spin>(
                "V0",
                "type=nucleus;spin=7/2;"
                "tensor=isotropic(1.0);");
        arma::vec field(3,arma::fill::zeros);
        field(2)=1.0;

        if (SpinAPI::NuclearZeeman::CreateInteraction(
                "NZ0",noIsotope,field,
                interaction,error) ||
            error !=
                "nuclear spin requires an explicit isotope property")
            return false;

        auto wrongSpin=
            std::make_shared<SpinAPI::Spin>(
                "Vwrong",
                "type=nucleus;spin=1/2;"
                "isotope=51V;"
                "tensor=isotropic(1.0);");
        if (SpinAPI::NuclearZeeman::CreateInteraction(
                "NZwrong",wrongSpin,field,
                interaction,error) ||
            error !=
                "nuclear isotope spin quantum number does not match Spin object")
            return false;

        auto nucleus=
            std::make_shared<SpinAPI::Spin>(
                "V",
                "type=nucleus;spin=7/2;"
                "isotope=51V;"
                "tensor=isotropic(1.0);");

        auto common=
            std::make_shared<SpinAPI::Interaction>(
                "NZcommon",
                "type=zeeman;spins=V;field=0 0 1;"
                "ignoretensors=true;"
                "commonprefactor=true;"
                "prefactor=1.0;");
        if (!common->ParseSpinGroups(
                std::vector<SpinAPI::spin_ptr>{nucleus}))
            return false;

        if (SpinAPI::NuclearZeeman::ValidateInteraction(
                nucleus,common,error) ||
            error !=
                "isotope-aware nuclear Zeeman must set commonprefactor=false")
            return false;

        auto wrongSign=
            std::make_shared<SpinAPI::Interaction>(
                "NZwrongSign",
                "type=zeeman;spins=V;field=0 0 1;"
                "ignoretensors=true;"
                "commonprefactor=false;"
                "prefactor=0.07045513257526152;");
        if (!wrongSign->ParseSpinGroups(
                std::vector<SpinAPI::spin_ptr>{nucleus}))
            return false;

        if (SpinAPI::NuclearZeeman::ValidateInteraction(
                nucleus,wrongSign,error) ||
            error !=
                "isotope-aware nuclear Zeeman prefactor does not match -g_N mu_N/hbar")
            return false;

        SpinAPI::NuclearIsotopeData unsupported;
        if (SpinAPI::NuclearIsotopeRegistry::Lookup(
                "50V",unsupported,error) ||
            error !=
                "unsupported nuclear isotope; authenticated registry currently contains only 51V")
            return false;

        return true;
    }

    bool GRC_TestNuclearZeemanHybridPreparation()
    {
        using namespace RunSection::General::Resonance;

        auto electron=
            std::make_shared<SpinAPI::Spin>(
                "E",
                "type=electron;spin=1/2;"
                "tensor=isotropic(2.0023);");
        auto nucleus=
            std::make_shared<SpinAPI::Spin>(
                "V",
                "type=nucleus;spin=7/2;"
                "isotope=51V;"
                "tensor=isotropic(1.0);");

        const double fieldT=0.34;

        auto b0=
            std::make_shared<SpinAPI::Interaction>(
                "B0",
                "type=zeeman;spins=E;"
                "field=0 0 0.34;"
                "ignoretensors=false;"
                "commonprefactor=true;"
                "prefactor=1.0;");
        auto hfc=
            std::make_shared<SpinAPI::Interaction>(
                "A",
                "type=hyperfine;"
                "group1=E;group2=V;"
                "tensor=isotropic(0.005);"
                "commonprefactor=false;"
                "prefactor=1.0;");

        arma::vec nuclearField(
            3,arma::fill::zeros);
        nuclearField(2)=fieldT;

        SpinAPI::interaction_ptr nz;
        std::string error;
        if (!SpinAPI::NuclearZeeman::CreateInteraction(
                "NZ",nucleus,nuclearField,
                nz,error))
            return false;

        auto up=
            std::make_shared<SpinAPI::State>(
                "Up","spin(E)=|1/2>;");

        auto system=
            std::make_shared<SpinAPI::SpinSystem>(
                "PhysicalV51");
        system->Add(electron);
        system->Add(nucleus);
        system->Add(b0);
        system->Add(hfc);
        system->Add(nz);
        system->Add(up);

        if (!system->ValidateInteractions().empty() ||
            !up->ParseFromSystem(*system) ||
            !SpinAPI::NuclearZeeman::
                ValidateInteraction(
                    nucleus,nz,error))
            return false;

        HybridNuclearResonancePartition partition;
        partition.system=system;
        partition.exactCoreSpins={electron};
        partition.exactCoreInteractions={b0};
        partition.exactCoreFieldInteractions={b0};
        partition.detectionTerms={{electron,b0}};
        HybridNuclearResonanceNucleusPartition factor;
        factor.nucleus=nucleus;
        factor.hyperfine=hfc;
        factor.nuclearInteractions={nz};
        factor.nuclearFieldInteractions={nz};
        partition.nuclei={factor};
        partition.exactCoreState=up;

        HybridNuclearResonancePoint point;
        if (!HybridNuclearResonancePreparation::
                BuildPoint(
                    partition,
                    GRC_IdentityOrientation(),
                    fieldT,point,error))
            return false;

        const arma::cx_mat iz=
            arma::conv_to<arma::cx_mat>::from(
                nucleus->Sz());
        const arma::cx_mat expectedDerivative=
            nz->Prefactor()*iz;
        const arma::cx_mat actualDerivative(
            point.hybrid.nuclei.front().nuclearDHdB);
        const arma::cx_mat expectedHamiltonian=
            fieldT*expectedDerivative;

        const double scale=std::max({
            1.0,
            arma::norm(expectedDerivative,"fro"),
            arma::norm(expectedHamiltonian,"fro")
        });

        return
            point.hybrid.nuclei.front().nuclearDimension==8 &&
            nz->Prefactor()<0.0 &&
            arma::norm(
                actualDerivative-
                expectedDerivative,"fro") <=
                1.0e-13*scale &&
            arma::norm(
                point.hybrid.nuclei.front().nuclearHamiltonian-
                expectedHamiltonian,"fro") <=
                1.0e-13*scale;
    }

    bool GRC_HybridPointNear(
        const RunSection::General::Resonance::HybridNuclearResonancePoint &a,
        const RunSection::General::Resonance::HybridNuclearResonancePoint &b,
        double tolerance)
    {
        const auto sparseNear=
            [&](const arma::sp_cx_mat &x,
                const arma::sp_cx_mat &y)
            {
                if (x.n_rows!=y.n_rows ||
                    x.n_cols!=y.n_cols)
                    return false;
                const arma::cx_mat xd(x),yd(y);
                const double scale=std::max({
                    1.0,
                    arma::norm(xd,"fro"),
                    arma::norm(yd,"fro")
                });
                return arma::norm(xd-yd,"fro") <=
                    tolerance*scale;
            };
        const auto denseNear=
            [&](const arma::cx_mat &x,
                const arma::cx_mat &y)
            {
                if (x.n_rows!=y.n_rows ||
                    x.n_cols!=y.n_cols)
                    return false;
                const double scale=std::max({
                    1.0,
                    arma::norm(x,"fro"),
                    arma::norm(y,"fro")
                });
                return arma::norm(x-y,"fro") <=
                    tolerance*scale;
            };

        return
            sparseNear(
                a.coreHamiltonian,b.coreHamiltonian) &&
            denseNear(
                a.coreDensity,b.coreDensity) &&
            sparseNear(
                a.coreDHdB,b.coreDHdB) &&
            denseNear(
                a.coreMuX,b.coreMuX) &&
            denseNear(
                a.coreMuY,b.coreMuY) &&
            denseNear(
                a.hybrid.nuclei.front().hyperfineCoreNuclear,
                b.hybrid.nuclei.front().hyperfineCoreNuclear) &&
            denseNear(
                a.hybrid.nuclei.front().nuclearHamiltonian,
                b.hybrid.nuclei.front().nuclearHamiltonian) &&
            sparseNear(
                a.hybrid.nuclei.front().nuclearDHdB,
                b.hybrid.nuclei.front().nuclearDHdB) &&
            a.hybrid.nuclei.front().nuclearDimension ==
                b.hybrid.nuclei.front().nuclearDimension &&
            std::abs(
                a.hybrid.nuclei.front().overlapThreshold-
                b.hybrid.nuclei.front().overlapThreshold) <=
                tolerance &&
            a.hybrid.nuclei.front().fieldIndependentProjection ==
                b.hybrid.nuclei.front().fieldIndependentProjection;
    }

    bool GRC_BuildZfsPreparedPoint(
        const GRC_ZfsHybridModel &model,
        const RunSection::General::HS::HSOrientation &orientation,
        double hyperfineScale,double fieldT,
        RunSection::General::Resonance::HybridNuclearResonancePoint &point,
        std::string &error)
    {
        using namespace RunSection::General::Resonance;

        auto system=
            GRC_BuildZfsHybridSystem(
                model,fieldT,hyperfineScale);
        if (system==nullptr)
        {
            error =
                "failed to build R2E-B ZFS system";
            return false;
        }

        auto electron=system->spins_find("E");
        auto nucleus=system->spins_find("V");
        auto b0=system->interactions_find("B0");
        auto zfs=system->interactions_find("ZFS");
        auto hfc=system->interactions_find("A");
        auto nz=system->interactions_find("NZ");
        auto up=system->states_find("Up");

        HybridNuclearResonancePartition partition;
        partition.system=system;
        partition.exactCoreSpins={electron};
        partition.exactCoreInteractions={b0,zfs};
        partition.exactCoreFieldInteractions={b0};
        partition.detectionTerms={{electron,b0}};
        HybridNuclearResonanceNucleusPartition factor;
        factor.nucleus=nucleus;
        factor.hyperfine=hfc;
        factor.nuclearInteractions={nz};
        factor.nuclearFieldInteractions={nz};
        partition.nuclei={factor};
        partition.exactCoreState=up;
        partition.fullTensorRotation=true;
        partition.nuclei.front().overlapThreshold=1.0e-14;
        partition.nuclei.front().fieldIndependentProjection=false;

        return HybridNuclearResonancePreparation::BuildPoint(
            partition,orientation,fieldT,point,error);
    }

    bool GRC_BuildExactCorePreparedPoint(
        const GRC_ExactCorePromotionModel &model,
        const RunSection::General::HS::HSOrientation &orientation,
        double perturbativeScale,double fieldT,
        bool swapCoreOrder,
        RunSection::General::Resonance::HybridNuclearResonancePoint &point,
        std::string &error)
    {
        using namespace RunSection::General::Resonance;

        auto system=
            GRC_BuildExactCorePromotionSystem(
                model,fieldT,perturbativeScale);
        if (system==nullptr)
        {
            error =
                "failed to build R2E-B exact-core system";
            return false;
        }

        auto electron=system->spins_find("E");
        auto exactNucleus=system->spins_find("Vexact");
        auto perturbativeNucleus=system->spins_find("Vpert");
        auto b0=system->interactions_find("B0");
        auto aExact=system->interactions_find("Aexact");
        auto aPert=system->interactions_find("Apert");
        auto nzExact=system->interactions_find("NZexact");
        auto nzPert=system->interactions_find("NZpert");
        auto up=system->states_find("Up");

        HybridNuclearResonancePartition partition;
        partition.system=system;
        partition.exactCoreSpins=swapCoreOrder
            ? std::vector<SpinAPI::spin_ptr>{
                exactNucleus,electron}
            : std::vector<SpinAPI::spin_ptr>{
                electron,exactNucleus};
        partition.exactCoreInteractions={
            b0,aExact,nzExact};
        partition.exactCoreFieldInteractions={
            b0,nzExact};
        partition.detectionTerms={{electron,b0}};
        HybridNuclearResonanceNucleusPartition factor;
        factor.nucleus=perturbativeNucleus;
        factor.hyperfine=aPert;
        factor.nuclearInteractions={nzPert};
        factor.nuclearFieldInteractions={nzPert};
        partition.nuclei={factor};
        partition.exactCoreState=up;
        partition.fullTensorRotation=true;
        partition.nuclei.front().overlapThreshold=1.0e-14;
        partition.nuclei.front().fieldIndependentProjection=false;

        return HybridNuclearResonancePreparation::BuildPoint(
            partition,orientation,fieldT,point,error);
    }

    bool GRC_TestHybridPreparationZfsPointParity()
    {
        GRC_ZfsHybridModel model;
        const auto orientation=
            GRC_Orientation(0.29,0.77,-0.41);
        const double frequency=9.5;
        const double fieldT=
            2.0*arma::datum::pi*frequency/
            (GRC_MU_B_OVER_HBAR*model.gz);

        RunSection::General::Resonance::
            HybridNuclearResonancePoint manual,prepared;
        std::string error;

        if (!GRC_BuildZfsHybridPoint(
                model,orientation,0.57,fieldT,
                manual,error) ||
            !GRC_BuildZfsPreparedPoint(
                model,orientation,0.57,fieldT,
                prepared,error))
            return false;

        return GRC_HybridPointNear(
            manual,prepared,1.0e-13);
    }

    bool GRC_TestHybridPreparationExactCorePointParity()
    {
        GRC_ExactCorePromotionModel model;
        const auto orientation=
            GRC_Orientation(-0.26,0.72,0.37);
        const double frequency=9.5;
        const double fieldT=
            2.0*arma::datum::pi*frequency/
            (GRC_MU_B_OVER_HBAR*model.gz);

        for (const bool swap:{false,true})
        {
            RunSection::General::Resonance::
                HybridNuclearResonancePoint manual,prepared;
            std::string error;

            if (!GRC_BuildExactCorePromotionPoint(
                    model,orientation,0.47,fieldT,
                    swap,manual,error) ||
                !GRC_BuildExactCorePreparedPoint(
                    model,orientation,0.47,fieldT,
                    swap,prepared,error))
                return false;

            if (!GRC_HybridPointNear(
                    manual,prepared,1.0e-13))
                return false;
        }

        return true;
    }

    bool GRC_LineSetsNear(
        const RunSection::General::Resonance::ResonanceLineSet &a,
        const RunSection::General::Resonance::ResonanceLineSet &b,
        double tolerance)
    {
        if (a.fieldJacobianQualified !=
                b.fieldJacobianQualified ||
            a.lines.size()!=b.lines.size())
            return false;

        for (std::size_t i=0;i<a.lines.size();++i)
        {
            const auto &x=a.lines[i];
            const auto &y=b.lines[i];

            if (x.lower!=y.lower ||
                x.upper!=y.upper)
                return false;

            const double scale=std::max({
                1.0,
                std::abs(x.omega),std::abs(y.omega),
                std::abs(x.dOmegaDB),std::abs(y.dOmegaDB),
                std::abs(x.moment.perpendicular),
                std::abs(y.moment.perpendicular)
            });

            if (std::abs(x.omega-y.omega)>tolerance*scale ||
                std::abs(
                    x.populationDifference-
                    y.populationDifference)>tolerance*scale ||
                std::abs(
                    x.dOmegaDB-y.dOmegaDB)>tolerance*scale ||
                std::abs(
                    x.dBdOmega-y.dBdOmega)>tolerance*scale ||
                std::abs(
                    x.moment.x-y.moment.x)>tolerance*scale ||
                std::abs(
                    x.moment.y-y.moment.y)>tolerance*scale ||
                std::abs(
                    x.moment.perpendicular-
                    y.moment.perpendicular)>tolerance*scale)
                return false;
        }

        return true;
    }

    bool GRC_TestHybridPreparationFieldResponseParity()
    {
        using namespace RunSection::General::Resonance;

        GRC_ZfsHybridModel model;
        const auto orientation=
            GRC_Orientation(0.33,0.75,-0.29);
        const double frequency=9.5;
        const double fieldT=
            2.0*arma::datum::pi*frequency/
            (GRC_MU_B_OVER_HBAR*model.gz);

        ResonanceLineSet manual,prepared;
        if (!GRC_BuildZfsHybridFiniteDifference(
                model,orientation,0.54,
                fieldT,frequency,1.0e-4,manual))
            return false;

        HybridNuclearResonancePointProvider provider=
            [&](double field,
                HybridNuclearResonancePoint &point,
                std::string &error)
            {
                return GRC_BuildZfsPreparedPoint(
                    model,orientation,0.54,
                    field,point,error);
            };

        HybridNuclearResonanceFieldResponseRequest response;
        response.fieldT=fieldT;
        response.fieldStepT=1.0e-4;
        response.minimumCoreStateOverlap=0.85;
        response.minimumNuclearStateOverlap=0.85;
        response.jacobianRelativeTolerance=5.0e-4;
        response.jacobianAbsoluteTolerance=2.0e-4;

        SpectrumRequest request;
        request.microwaveFrequencyGHz=frequency;
        request.linewidth_mT=0.10;
        request.populationThreshold=1.0e-15;
        request.minimumSlope=1.0e-15;

        std::string error;
        if (!GRC_GenerateFirstOrderFiniteDifference(
                    provider,response,request,
                    prepared,error))
            return false;

        return GRC_LineSetsNear(
            manual,prepared,1.0e-12);
    }

    bool GRC_TestHybridPreparationFailsClosed()
    {
        using namespace RunSection::General::Resonance;

        GRC_ZfsHybridModel model;
        const auto orientation=
            GRC_Orientation(-0.21,0.69,0.31);
        const double frequency=9.5;
        const double fieldT=
            2.0*arma::datum::pi*frequency/
            (GRC_MU_B_OVER_HBAR*model.gz);

        auto system=
            GRC_BuildZfsHybridSystem(
                model,fieldT,0.50);
        if (system==nullptr)
            return false;

        auto electron=system->spins_find("E");
        auto nucleus=system->spins_find("V");
        auto b0=system->interactions_find("B0");
        auto zfs=system->interactions_find("ZFS");
        auto hfc=system->interactions_find("A");
        auto nz=system->interactions_find("NZ");
        auto up=system->states_find("Up");

        HybridNuclearResonancePartition valid;
        valid.system=system;
        valid.exactCoreSpins={electron};
        valid.exactCoreInteractions={b0,zfs};
        valid.exactCoreFieldInteractions={b0};
        valid.detectionTerms={{electron,b0}};
        HybridNuclearResonanceNucleusPartition factor;
        factor.nucleus=nucleus;
        factor.hyperfine=hfc;
        factor.nuclearInteractions={nz};
        factor.nuclearFieldInteractions={nz};
        valid.nuclei={factor};
        valid.exactCoreState=up;

        HybridNuclearResonancePoint point;
        std::string error;

        auto duplicateSpin=valid;
        duplicateSpin.exactCoreSpins.push_back(nucleus);
        if (HybridNuclearResonancePreparation::BuildPoint(
                duplicateSpin,orientation,fieldT,
                point,error) ||
            error !=
                "hybrid partition perturbative nucleus must not be part of the exact core")
            return false;

        auto unowned=valid;
        unowned.nuclei.front().nuclearInteractions.clear();
        unowned.nuclei.front().nuclearFieldInteractions.clear();
        if (HybridNuclearResonancePreparation::BuildPoint(
                unowned,orientation,fieldT,
                point,error) ||
            error !=
                "hybrid partition must own every SpinSystem interaction exactly once")
            return false;

        auto polarized=std::make_shared<SpinAPI::State>(
            "Polarized",
            "spin(E)=|1/2>;spin(V)=|7/2>;");
        if (!system->Add(polarized) ||
            !polarized->ParseFromSystem(*system))
            return false;

        auto polarizedPartition=valid;
        polarizedPartition.exactCoreState=polarized;
        if (HybridNuclearResonancePreparation::BuildPoint(
                polarizedPartition,orientation,fieldT,
                point,error) ||
            error !=
                "hybrid perturbative nucleus reference state must be unpolarized and unspecified")
            return false;

        return true;
    }

    bool GRC_TestMagneticMomentBuilderMatchesSpinAPIZeemanAxes()
    {
        using namespace RunSection::General::Resonance;

        auto electron = std::make_shared<SpinAPI::Spin>(
            "E",
            "type=electron;spin=1/2;"
            "tensor=anisotropic(1.91 2.07 2.31);");

        const std::string frame =
            "orientation=0.27,0.58,-0.31;"
            "ignoretensors=false;"
            "commonprefactor=true;prefactor=1.0;";

        auto b0 = std::make_shared<SpinAPI::Interaction>(
            "B0",
            "type=zeeman;spins=E;field=0 0 0.34;" +
            frame);
        auto bx = std::make_shared<SpinAPI::Interaction>(
            "Bx",
            "type=zeeman;spins=E;field=1 0 0;" +
            frame);
        auto by = std::make_shared<SpinAPI::Interaction>(
            "By",
            "type=zeeman;spins=E;field=0 1 0;" +
            frame);

        auto system = std::make_shared<SpinAPI::SpinSystem>(
            "MomentAxes");
        system->Add(electron);
        system->Add(b0);
        system->Add(bx);
        system->Add(by);
        if (!system->ValidateInteractions().empty())
            return false;

        SpinAPI::SpinSpace space(
            std::vector<SpinAPI::spin_ptr>{electron});
        space.UseSuperoperatorSpace(false);
        space.UseFullTensorRotation(true);

        const auto orientation =
            GRC_Orientation(0.41,0.73,-0.22);

        arma::cx_mat muX,muY;
        std::string error;
        if (!ResonanceMagneticMomentBuilder::BuildTransverse(
                space,{{electron,b0}},
                orientation.frameToLab,true,
                muX,muY,error))
            return false;

        arma::mat rotation=orientation.frameToLab;
        arma::sp_cx_mat hx,hy;
        if (!space.InteractionOperatorRotatedZYZ(
                bx,rotation,hx) ||
            !space.InteractionOperatorRotatedZYZ(
                by,rotation,hy))
            return false;

        const arma::cx_mat hxDense(hx);
        const arma::cx_mat hyDense(hy);
        const double scale=std::max({
            1.0,arma::norm(hxDense,"fro"),
            arma::norm(hyDense,"fro")
        });

        return
            arma::norm(muX-hxDense,"fro") <= 1.0e-13*scale &&
            arma::norm(muY-hyDense,"fro") <= 1.0e-13*scale;
    }

    bool GRC_TestMagneticMomentBuilderOrientationContract()
    {
        using namespace RunSection::General::Resonance;

        auto electron = std::make_shared<SpinAPI::Spin>(
            "E",
            "type=electron;spin=1/2;"
            "tensor=anisotropic(1.88 2.11 2.37);");
        auto anisotropic = std::make_shared<SpinAPI::Interaction>(
            "B0",
            "type=zeeman;spins=E;field=0 0 0.34;"
            "orientation=-0.19,0.62,0.28;"
            "ignoretensors=false;"
            "commonprefactor=true;prefactor=1.0;");
        auto ignoreTensor = std::make_shared<SpinAPI::Interaction>(
            "Biso",
            "type=zeeman;spins=E;field=0 0 0.34;"
            "orientation=-0.19,0.62,0.28;"
            "ignoretensors=true;"
            "commonprefactor=true;prefactor=1.0;");

        auto system = std::make_shared<SpinAPI::SpinSystem>(
            "MomentOrientation");
        system->Add(electron);
        system->Add(anisotropic);
        system->Add(ignoreTensor);
        if (!system->ValidateInteractions().empty())
            return false;

        SpinAPI::SpinSpace space(
            std::vector<SpinAPI::spin_ptr>{electron});
        space.UseSuperoperatorSpace(false);
        space.UseFullTensorRotation(true);

        const arma::mat identity =
            arma::eye<arma::mat>(3,3);
        const auto orientation =
            GRC_Orientation(-0.36,0.81,0.47);

        arma::cx_mat ax0,ay0,ax1,ay1;
        arma::cx_mat ix0,iy0,ix1,iy1;
        std::string error;

        if (!ResonanceMagneticMomentBuilder::BuildTransverse(
                space,{{electron,anisotropic}},
                identity,true,ax0,ay0,error) ||
            !ResonanceMagneticMomentBuilder::BuildTransverse(
                space,{{electron,anisotropic}},
                orientation.frameToLab,true,
                ax1,ay1,error) ||
            !ResonanceMagneticMomentBuilder::BuildTransverse(
                space,{{electron,ignoreTensor}},
                identity,true,ix0,iy0,error) ||
            !ResonanceMagneticMomentBuilder::BuildTransverse(
                space,{{electron,ignoreTensor}},
                orientation.frameToLab,true,
                ix1,iy1,error))
            return false;

        const double anisotropicShift =
            arma::norm(ax1-ax0,"fro") +
            arma::norm(ay1-ay0,"fro");
        const double isotropicScale=std::max({
            1.0,
            arma::norm(ix0,"fro"),
            arma::norm(iy0,"fro")
        });

        return anisotropicShift > 1.0e-3 &&
            arma::norm(ix1-ix0,"fro") <=
                1.0e-13*isotropicScale &&
            arma::norm(iy1-iy0,"fro") <=
                1.0e-13*isotropicScale;
    }

    bool GRC_TestMagneticMomentBuilderFailsClosedOnOwnership()
    {
        using namespace RunSection::General::Resonance;

        auto e1 = std::make_shared<SpinAPI::Spin>(
            "E1","type=electron;spin=1/2;"
            "tensor=isotropic(2.0);");
        auto e2 = std::make_shared<SpinAPI::Spin>(
            "E2","type=electron;spin=1/2;"
            "tensor=isotropic(2.0);");
        auto b1 = std::make_shared<SpinAPI::Interaction>(
            "B1",
            "type=zeeman;spins=E1;field=0 0 0.34;"
            "ignoretensors=false;"
            "commonprefactor=true;prefactor=1.0;");

        auto system = std::make_shared<SpinAPI::SpinSystem>(
            "MomentOwnership");
        system->Add(e1);
        system->Add(e2);
        system->Add(b1);
        if (!system->ValidateInteractions().empty())
            return false;

        SpinAPI::SpinSpace space(
            std::vector<SpinAPI::spin_ptr>{e1});
        space.UseSuperoperatorSpace(false);

        arma::cx_mat muX,muY;
        std::string error;

        if (ResonanceMagneticMomentBuilder::BuildTransverse(
                space,{{e2,b1}},
                arma::eye<arma::mat>(3,3),true,
                muX,muY,error))
            return false;
        if (error !=
            "resonance magnetic-moment detection spin is outside the supplied SpinSpace")
            return false;

        if (ResonanceMagneticMomentBuilder::BuildTransverse(
                space,{{e1,nullptr}},
                arma::eye<arma::mat>(3,3),true,
                muX,muY,error))
            return false;

        return error ==
            "resonance magnetic-moment term contains a null spin or interaction";
    }

    bool GRC_TestExactLineBackendSeam()
    {
        using namespace RunSection::General::Resonance;

        GRC_Model model{2.0023,2.0023,2.0023,0.060};
        const double frequency = 9.5;
        const double omega = 2.0*arma::datum::pi*frequency;
        const double fieldT = omega/(GRC_MU_B_OVER_HBAR*model.gz);

        auto system = GRC_BuildSystem(model,fieldT);
        if (system == nullptr) return false;
        SpinAPI::SpinSpace space(*system);
        space.UseSuperoperatorSpace(false);
        space.UseFullTensorRotation(true);

        const auto plan = GRC_Plan(model,false);
        const auto orientation = GRC_IdentityOrientation();
        GeneralResonanceHamiltonian builder(plan,space);

        arma::sp_cx_mat H,dHdB;
        arma::cx_mat rho,muX,muY;
        std::string error;
        if (!builder.Build(orientation,H,error) ||
            !builder.BuildFieldDerivative(orientation,{"B0"},fieldT,dHdB,error) ||
            !GRC_Density(system,space,rho) ||
            !GRC_TransverseOperators(system,space,muX,muY))
            return false;

        SpectrumRequest request;
        request.microwaveFrequencyGHz = frequency;
        request.linewidth_mT = 0.12;
        request.lineshape = Lineshape::Gaussian;

        SpectrumPoint compatibility;
        if (!ResonanceSpectrumEvaluator::Evaluate(
                H,rho,dHdB,muX,muY,request,compatibility,error))
            return false;

        ResonanceLineSet lines;
        if (!ExactResonanceSolver::Generate(
                H,rho,dHdB,muX,muY,request,lines,error))
            return false;
        if (lines.lines.empty() || !lines.fieldJacobianQualified)
            return false;

        // Backend-neutral lines must not depend on the experimental
        // microwave frequency. Only spectrum assembly applies detuning.
        SpectrumRequest otherFrequency = request;
        otherFrequency.microwaveFrequencyGHz = 94.0;
        ResonanceLineSet linesW;
        if (!ExactResonanceSolver::Generate(
                H,rho,dHdB,muX,muY,otherFrequency,linesW,error))
            return false;
        if (linesW.lines.size() != lines.lines.size() ||
            !linesW.fieldJacobianQualified)
            return false;
        for (size_t i=0; i<lines.lines.size(); ++i)
        {
            const auto &a = lines.lines[i];
            const auto &b = linesW.lines[i];
            if (a.lower != b.lower || a.upper != b.upper ||
                a.omega != b.omega ||
                a.populationDifference != b.populationDifference ||
                a.dOmegaDB != b.dOmegaDB ||
                a.dBdOmega != b.dBdOmega ||
                a.moment.x != b.moment.x ||
                a.moment.y != b.moment.y ||
                a.moment.perpendicular != b.moment.perpendicular)
                return false;
        }

        SpectrumPoint explicitLinePath;
        if (!ResonanceSpectrumEvaluator::Evaluate(
                lines,request,explicitLinePath,error))
            return false;

        const double scale = std::max({
            1.0,
            std::abs(compatibility.totalX),
            std::abs(compatibility.totalY),
            std::abs(compatibility.totalPerpendicular)
        });

        return compatibility.acceptedTransitions ==
                   explicitLinePath.acceptedTransitions &&
               std::abs(compatibility.totalX-explicitLinePath.totalX) <=
                   1.0e-14*scale &&
               std::abs(compatibility.totalY-explicitLinePath.totalY) <=
                   1.0e-14*scale &&
               std::abs(compatibility.totalPerpendicular-
                        explicitLinePath.totalPerpendicular) <=
                   1.0e-14*scale;
    }

    bool GRC_TestLegacyParityIsotropicG()
    {
        GRC_Model model;
        const double frequency=9.5;
        const double center=(2.0*arma::datum::pi*frequency)/(GRC_MU_B_OVER_HBAR*model.gz);
        return GRC_CompareLegacyAndGeneral(model,center,0.0015,0.0001,frequency,0.20);
    }

    bool GRC_TestLegacyParityAxialG()
    {
        GRC_Model model{2.0,2.0,2.4,0.0};
        const double frequency=9.5;
        const double center=(2.0*arma::datum::pi*frequency)/(GRC_MU_B_OVER_HBAR*model.gz);
        return GRC_CompareLegacyAndGeneral(model,center,0.0015,0.0001,frequency,0.20);
    }

    bool GRC_TestLegacyParityHyperfine()
    {
        GRC_Model model{2.0023,2.0023,2.0023,0.060};
        const double frequency=9.5;
        const double center=(2.0*arma::datum::pi*frequency)/(GRC_MU_B_OVER_HBAR*model.gz);
        return GRC_CompareLegacyAndGeneral(model,center,0.0010,0.00005,frequency,0.12);
    }
}

void AddGeneralResonanceCoreTests(std::vector<test_case> &cases)
{
    cases.push_back(test_case("General resonance core normalized Gaussian/Lorentzian FWHM contract",GRC_TestLineshapeContract));
    cases.push_back(test_case("General resonance core field Jacobian and transition detection",GRC_TestFieldJacobianAndDetector));
    cases.push_back(test_case("General resonance core resolves degenerate field slopes",GRC_TestDegenerateFieldJacobian));
    cases.push_back(test_case("General resonance Hamiltonian adapter preserves full/secular distinction",GRC_TestHamiltonianAdapterPreservesApproximation));
    cases.push_back(test_case("General resonance R2I resolved detection builder sum contract",GRC_TestR2IBuilderResolvedChannelSum));
    cases.push_back(test_case("General resonance R2I transition-moment channel decomposition",GRC_TestR2ITransitionMomentDecomposition));
    cases.push_back(test_case("General resonance R2I exact resolved-line total parity",GRC_TestR2IExactResolvedLineParity));
    cases.push_back(test_case("General resonance R2I hybrid resolved-channel scaling",GRC_TestR2IHybridResolvedScaling));
    cases.push_back(test_case("General resonance R2I merged spectator channel conservation",GRC_TestR2IMergedSpectatorChannelConservation));
    cases.push_back(test_case("General resonance R2H canonical one/multi cardinality contract",GRC_TestR2HCanonicalCardinalityContract));
    cases.push_back(test_case("General resonance R2H physical two-51V canonical point parity",GRC_TestR2HCanonicalTwoV51PointParity));
    cases.push_back(test_case("General resonance R2H physical two-51V canonical field-response parity",GRC_TestR2HCanonicalTwoV51FieldResponseParity));
    cases.push_back(test_case("General resonance R2H canonical partition ownership fails closed",GRC_TestR2HCanonicalOwnershipFailsClosed));
    cases.push_back(test_case("General resonance R2G-B multi-nucleus N=1 finite-difference parity",GRC_TestR2GBMultiNucleusN1FieldResponseParity));
    cases.push_back(test_case("General resonance R2G-B multi-nucleus field-response permutation invariance",GRC_TestR2GBMultiNucleusPermutationInvariant));
    cases.push_back(test_case("General resonance R2G-B multi-nucleus finite-difference step convergence",GRC_TestR2GBMultiNucleusStepConvergence));
    cases.push_back(test_case("General resonance R2G-B merged field-response branches fail closed",GRC_TestR2GBMultiNucleusMergingFailsClosed));
    cases.push_back(test_case("General resonance R2G-B conditional two-I=1/2 exact field-response parity",GRC_TestR2GBConditionalTwoI12ExactFieldResponseParity));
    cases.push_back(test_case("General resonance R2G-A independent multi-nucleus N=1 parity",GRC_TestR2GAMultiNucleusN1Parity));
    cases.push_back(test_case("General resonance R2G-A independent multi-nucleus permutation invariance",GRC_TestR2GAMultiNucleusPermutationInvariant));
    cases.push_back(test_case("General resonance R2G-A spectator nucleus and transition-weight conservation",GRC_TestR2GAMultiNucleusSpectatorAndWeightConservation));
    cases.push_back(test_case("General resonance R2G-A component controls fail closed and report pruning",GRC_TestR2GAMultiNucleusControlsFailClosed));
    cases.push_back(test_case("General resonance R2G-A two-I=1/2 exact residual retains A^2 scaling",GRC_TestR2GAMultiNucleusExactA2Scaling));
    cases.push_back(test_case("General resonance R2F 51V isotope registry and bare nuclear gamma",GRC_TestNuclearZeemanV51Constants));
    cases.push_back(test_case("General resonance R2F isotope-aware nuclear Zeeman signed Hamiltonian",GRC_TestNuclearZeemanSignedHamiltonian));
    cases.push_back(test_case("General resonance R2F isotope-aware nuclear Zeeman fails closed",GRC_TestNuclearZeemanFailsClosed));
    cases.push_back(test_case("General resonance R2F physical 51V nuclear Zeeman hybrid preparation",GRC_TestNuclearZeemanHybridPreparation));
    cases.push_back(test_case("General resonance R2E-B ZFS hybrid partition point parity",GRC_TestHybridPreparationZfsPointParity));
    cases.push_back(test_case("General resonance R2E-B promoted exact-core partition point parity",GRC_TestHybridPreparationExactCorePointParity));
    cases.push_back(test_case("General resonance R2E-B partition finite-difference solver parity",GRC_TestHybridPreparationFieldResponseParity));
    cases.push_back(test_case("General resonance R2E-B partition ownership fails closed",GRC_TestHybridPreparationFailsClosed));
    cases.push_back(test_case("General resonance magnetic moment matches oriented SpinAPI Zeeman x/y derivatives",GRC_TestMagneticMomentBuilderMatchesSpinAPIZeemanAxes));
    cases.push_back(test_case("General resonance magnetic moment orientation contract",GRC_TestMagneticMomentBuilderOrientationContract));
    cases.push_back(test_case("General resonance magnetic moment ownership fails closed",GRC_TestMagneticMomentBuilderFailsClosedOnOwnership));
    cases.push_back(test_case("General resonance exact solver line-backend seam parity",GRC_TestExactLineBackendSeam));
    cases.push_back(test_case("General resonance R2A one-I=7/2 hybrid zero-coupling limit",GRC_TestHybridI72ZeroLimit));
    cases.push_back(test_case("General resonance R2A one-I=7/2 first-order A^2 error scaling",GRC_TestHybridI72FirstOrderScaling));
    cases.push_back(test_case("General resonance R2A one-I=7/2 perturbation improves from X to W band",GRC_TestHybridI72HighFieldImproves));
    cases.push_back(test_case("General resonance R2A unqualified hybrid Jacobian fails closed",GRC_TestHybridUnqualifiedJacobianFailsClosed));
    cases.push_back(test_case("General resonance R2A anisotropic non-coaxial A^2 scaling at arbitrary orientation",GRC_TestHybridAnisotropicA2ScalingAtArbitraryOrientation));
    cases.push_back(test_case("General resonance R2A anisotropic perturbation improves from X to W band",GRC_TestHybridAnisotropicXWImprovement));
    cases.push_back(test_case("General resonance R2A non-coaxial hyperfine tensor frame is active",GRC_TestHybridNonCoaxialTensorFrameIsActive));
    cases.push_back(test_case("General resonance R2A rigid z-frame/powder composition covariance",GRC_TestHybridRigidZFramePowderComposition));
    cases.push_back(test_case("General resonance R2C S=1 ZFS-aware finite-difference hybrid field response",GRC_TestHybridZfsFieldResponseS1));
    cases.push_back(test_case("General resonance R2C ZFS-aware field-response error retains A^2 scaling",GRC_TestHybridZfsFieldResponseA2Scaling));
    cases.push_back(test_case("General resonance R2C finite-difference hybrid Jacobian step convergence",GRC_TestHybridZfsFieldResponseStepConvergence));
    cases.push_back(test_case("General resonance R2C S=3/2 ZFS-aware hybrid field response",GRC_TestHybridZfsFieldResponseS32));
    cases.push_back(test_case("General resonance R2D exact-core I=7/2 promotion zero-perturbative parity",GRC_TestHybridExactCorePromotionZeroPerturbativeParity));
    cases.push_back(test_case("General resonance R2D exact-core basis-order invariance",GRC_TestHybridExactCorePromotionBasisOrderInvariant));
    cases.push_back(test_case("General resonance R2D exact-core promotion requires corrected field response",GRC_TestHybridExactCorePromotionFieldResponse));
    cases.push_back(test_case("General resonance R2D exact-core promotion residual A^2 scaling",GRC_TestHybridExactCorePromotionA2Scaling));
    cases.push_back(test_case("General resonance core frozen legacy isotropic-g sweep parity",GRC_TestLegacyParityIsotropicG));
    cases.push_back(test_case("General resonance core frozen legacy axial-g sweep parity",GRC_TestLegacyParityAxialG));
    cases.push_back(test_case("General resonance core frozen legacy hyperfine sweep parity",GRC_TestLegacyParityHyperfine));
}
