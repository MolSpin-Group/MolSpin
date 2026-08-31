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
#include "ResonanceFieldJacobian.h"
#include "ResonanceLineshape.h"
#include "ResonanceSpectrumEvaluator.h"
#include "ResonanceTransitionDetector.h"
#include "ResonanceTransitionMoments.h"
#include "HSHamiltonianBuilder.h"
#include "Interaction.h"
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

    bool GRC_Density(const SpinAPI::system_ptr &system, SpinAPI::SpinSpace &space, arma::cx_mat &rho)
    {
        auto state = system->states_find("Up");
        if (state == nullptr || !space.GetState(state, rho)) return false;
        const arma::cx_double tr = arma::trace(rho);
        if (std::abs(tr) < 1.0e-15) return false;
        rho /= tr;
        return rho.is_finite();
    }

    bool GRC_TransverseOperators(const SpinAPI::system_ptr &system, SpinAPI::SpinSpace &space,
                                 arma::cx_mat &muX, arma::cx_mat &muY)
    {
        auto spin = system->spins_find("E");
        auto zeeman = system->interactions_find("B0");
        if (spin == nullptr || zeeman == nullptr) return false;

        arma::cx_mat Sx, Sy, Sz;
        if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sx()), spin, Sx) ||
            !space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sy()), spin, Sy) ||
            !space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sz()), spin, Sz))
            return false;

        arma::mat g = zeeman->IgnoreTensors()
            ? arma::eye<arma::mat>(3,3)
            : arma::conv_to<arma::mat>::from(spin->GetTensor().LabFrame());
        double prefactor = zeeman->Prefactor();
        if (zeeman->AddCommonPrefactor()) prefactor *= GRC_MU_B_OVER_HBAR;

        muX = prefactor * (g(0,0)*Sx + g(1,0)*Sy + g(2,0)*Sz);
        muY = prefactor * (g(0,1)*Sx + g(1,1)*Sy + g(2,1)*Sz);
        return muX.is_finite() && muY.is_finite();
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
            if (!GRC_Density(system, space, rho) || !GRC_TransverseOperators(system, space, muX, muY))
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

        OneNucleusHybridRequest hybrid;
        hybrid.hyperfineCoreNuclear = arma::cx_mat(hyperfinePair);
        hybrid.nuclearHamiltonian = arma::cx_mat(nuclearH);
        hybrid.nuclearDHdB = nuclearH/fieldT;
        hybrid.nuclearDimension =
            static_cast<arma::uword>(nucleus->Multiplicity());
        hybrid.overlapThreshold = 1.0e-14;
        hybrid.fieldIndependentProjection = qualifyProjection;

        return HybridNuclearResonanceSolver::GenerateFirstOrder(
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
                system,fullSpace,fullMuX,fullMuY) ||
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
                system,coreSpace,coreMuX,coreMuY) ||
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

        OneNucleusHybridRequest hybrid;
        hybrid.hyperfineCoreNuclear = arma::cx_mat(hyperfinePair);
        hybrid.nuclearHamiltonian = arma::cx_mat(nuclearH);
        hybrid.nuclearDHdB = nuclearH/fieldT;
        hybrid.nuclearDimension =
            static_cast<arma::uword>(nucleus->Multiplicity());
        hybrid.overlapThreshold = 1.0e-14;
        hybrid.fieldIndependentProjection = true;

        return HybridNuclearResonanceSolver::GenerateFirstOrder(
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
        RunSection::General::Resonance::OneNucleusHybridPoint &point,
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
                system,coreSpace,
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

        point.hybrid.hyperfineCoreNuclear =
            arma::cx_mat(hyperfinePair);
        point.hybrid.nuclearHamiltonian =
            arma::cx_mat(nuclearH);
        point.hybrid.nuclearDHdB =
            nuclearH/fieldT;
        point.hybrid.nuclearDimension =
            static_cast<arma::uword>(
                nucleus->Multiplicity());
        point.hybrid.overlapThreshold=1.0e-14;
        point.hybrid.fieldIndependentProjection=false;
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
                system,fullSpace,muX,muY))
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

    double GRC_StrongSlopeError(
        const RunSection::General::Resonance::ResonanceLineSet &reference,
        const RunSection::General::Resonance::ResonanceLineSet &candidate,
        std::size_t count)
    {
        // Validation-only branch identity:
        //   1) select the strongest exact reference transitions;
        //   2) retain hybrid candidates with comparable transition strength;
        //   3) choose the unique nearest-frequency candidate;
        //   4) compare dOmega/dB only after branch identity is established.
        //
        // We do not use dOmega/dB in matching because it is the quantity
        // under validation. The multiplicative weight window is symmetric:
        // 1/1.25 = 0.8. A branch outside this generous window is not treated
        // as the same strong transition in this perturbative validation gate.
        constexpr double minimumWeightRatio = 0.80;
        constexpr double maximumWeightRatio = 1.25;
        constexpr double maximumFrequencyMismatch = 5.0e-2;
        constexpr double ambiguityTolerance = 1.0e-8;

        const auto referenceLines =
            GRC_StrongestSlopeLines(reference,count);
        if (referenceLines.size()!=count)
            return -1.0;

        std::vector<GRC_WeightedSlopeLine> candidateLines;
        for (const auto &line:candidate.lines)
        {
            const double weight=GRC_LineWeight(line);
            if (weight>1.0e-16)
                candidateLines.push_back(
                    {weight,line.omega,line.dOmegaDB});
        }
        if (candidateLines.size()<count)
            return -1.0;

        std::vector<std::size_t> used;
        used.reserve(count);

        double result=0.0;
        for (const auto &referenceLine:referenceLines)
        {
            double bestDifference=
                std::numeric_limits<double>::infinity();
            double secondDifference=
                std::numeric_limits<double>::infinity();
            std::size_t bestIndex=0;
            bool found=false;

            for (std::size_t j=0;j<candidateLines.size();++j)
            {
                if (std::find(used.begin(),used.end(),j)!=used.end())
                    continue;

                const double weightRatio=
                    candidateLines[j].weight/referenceLine.weight;
                if (weightRatio<minimumWeightRatio ||
                    weightRatio>maximumWeightRatio)
                    continue;

                const double difference=
                    std::abs(
                        candidateLines[j].omega-referenceLine.omega);

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
                bestDifference>maximumFrequencyMismatch)
                return -1.0;

            if (std::isfinite(secondDifference) &&
                secondDifference-bestDifference<ambiguityTolerance)
                return -1.0;

            used.push_back(bestIndex);
            result=std::max(
                result,
                std::abs(
                    referenceLine.slope-
                    candidateLines[bestIndex].slope));
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

        OneNucleusHybridPointProvider provider =
            [&](double field,
                OneNucleusHybridPoint &point,
                std::string &error)
            {
                return GRC_BuildZfsHybridPoint(
                    model,orientation,
                    hyperfineScale,field,point,error);
            };

        OneNucleusHybridFieldResponseRequest response;
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
        return HybridNuclearResonanceSolver::
            GenerateFirstOrderFiniteDifference(
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

        OneNucleusHybridPoint center;
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
        if (!HybridNuclearResonanceSolver::GenerateFirstOrder(
                center.coreHamiltonian,
                center.coreDensity,
                center.coreDHdB,
                center.coreMuX,
                center.coreMuY,
                center.hybrid,
                request,uncorrected,error))
            return false;
        if (uncorrected.fieldJacobianQualified)
            return false;

        const double rawError=
            GRC_StrongSlopeError(exact,uncorrected,8);
        const double correctedError=
            GRC_StrongSlopeError(
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
            GRC_StrongSlopeError(
                exactStrong,hybridStrong,8);
        const double weakError=
            GRC_StrongSlopeError(
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

        OneNucleusHybridPoint center;
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
        if (!HybridNuclearResonanceSolver::GenerateFirstOrder(
                center.coreHamiltonian,
                center.coreDensity,
                center.coreDHdB,
                center.coreMuX,
                center.coreMuY,
                center.hybrid,
                request,uncorrected,error))
            return false;

        const double rawError=
            GRC_StrongSlopeError(exact,uncorrected,8);
        const double correctedError=
            GRC_StrongSlopeError(exact,corrected,8);

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
        RunSection::General::Resonance::OneNucleusHybridPoint &point,
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
                system,coreSpace,
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

        point.hybrid.hyperfineCoreNuclear =
            arma::cx_mat(hyperfinePair);
        point.hybrid.nuclearHamiltonian =
            arma::cx_mat(nuclearH);
        point.hybrid.nuclearDHdB =
            nuclearH/fieldT;
        point.hybrid.nuclearDimension =
            static_cast<arma::uword>(
                perturbativeNucleus->Multiplicity());
        point.hybrid.overlapThreshold = 1.0e-14;

        // The exact core contains a strongly coupled nucleus, so its
        // eigenvectors are field-dependent even for S=1/2.
        point.hybrid.fieldIndependentProjection = false;

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
                system,fullSpace,muX,muY))
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

        OneNucleusHybridPointProvider provider =
            [&](double field,
                OneNucleusHybridPoint &point,
                std::string &error)
            {
                return GRC_BuildExactCorePromotionPoint(
                    model,orientation,perturbativeScale,
                    field,swapCoreOrder,point,error);
            };

        OneNucleusHybridFieldResponseRequest response;
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
        return HybridNuclearResonanceSolver::
            GenerateFirstOrderFiniteDifference(
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

        OneNucleusHybridPoint center;
        std::string error;
        if (!GRC_BuildExactCorePromotionPoint(
                model,orientation,0.0,fieldT,false,
                center,error))
            return false;

        if (center.coreHamiltonian.n_rows != 16 ||
            center.coreHamiltonian.n_cols != 16 ||
            center.hybrid.nuclearDimension != 8 ||
            center.hybrid.hyperfineCoreNuclear.n_rows != 128 ||
            center.hybrid.hyperfineCoreNuclear.n_cols != 128)
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

        OneNucleusHybridPoint center;
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
        if (!HybridNuclearResonanceSolver::GenerateFirstOrder(
                center.coreHamiltonian,
                center.coreDensity,
                center.coreDHdB,
                center.coreMuX,
                center.coreMuY,
                center.hybrid,
                request,incomplete,error))
            return false;

        if (incomplete.fieldJacobianQualified ||
            !corrected.fieldJacobianQualified)
            return false;

        const double rawError =
            GRC_StrongSlopeError(exact,incomplete,8);
        const double correctedError =
            GRC_StrongSlopeError(exact,corrected,8);

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
            GRC_StrongSlopeError(
                exactStrong,hybridStrong,8);
        const double weakError =
            GRC_StrongSlopeError(
                exactWeak,hybridWeak,8);

        if (!(strongError>0.0) ||
            !(weakError>0.0))
            return false;

        const double ratio=strongError/weakError;
        return ratio>2.8 && ratio<5.5;
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
