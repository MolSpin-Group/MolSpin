//////////////////////////////////////////////////////////////////////////////
// General spectroscopy qualification tests.
//
// These tests qualify the already-generalized Hamiltonian/orientation contracts
// against analytic magnetic-resonance invariants and the frozen standalone
// resonance task. They intentionally do not introduce a fourth execution
// backend and do not modify historical spectroscopy implementations.
//////////////////////////////////////////////////////////////////////////////
#include "../RunSection/General/GeneralOrientationSampler.h"
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
    constexpr double GSQ_MU_B_OVER_HBAR = 8.79410005e+1; // rad ns^-1 T^-1 for g=1

    std::string GSQ_MatrixString(const arma::mat &m)
    {
        std::ostringstream out;
        out << std::setprecision(17);
        for (arma::uword r = 0; r < m.n_rows; ++r)
        {
            if (r != 0) out << ";";
            for (arma::uword c = 0; c < m.n_cols; ++c)
            {
                if (c != 0) out << " ";
                out << m(r,c);
            }
        }
        return out.str();
    }

    SpinAPI::system_ptr GSQ_BuildOneElectron(double gx,double gy,double gz,double fieldT)
    {
        std::ostringstream spinProps;
        spinProps << std::setprecision(17)
                  << "type=electron;spin=1/2;tensor=anisotropic("
                  << gx << " " << gy << " " << gz << ");";
        auto e = std::make_shared<SpinAPI::Spin>("E",spinProps.str());

        std::ostringstream zProps;
        zProps << std::setprecision(17)
               << "type=zeeman;spins=E;field=0 0 " << fieldT
               << ";ignoretensors=false;commonprefactor=true;prefactor=1.0;";
        auto z = std::make_shared<SpinAPI::Interaction>("B0",zProps.str());
        auto up = std::make_shared<SpinAPI::State>("Up","spin(E)=|1/2>;");

        auto system = std::make_shared<SpinAPI::SpinSystem>("System");
        system->Add(e); system->Add(z); system->Add(up);
        system->SetProperties(std::make_shared<MSDParser::ObjectParser>("properties","initialstate=Up;"));
        if (!system->ValidateInteractions().empty()) return nullptr;
        if (!up->ParseFromSystem(*system)) return nullptr;
        return system;
    }

    RunSection::General::HS::HSExecutionPlan GSQ_MakePlan(bool secular)
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
        return plan;
    }

    RunSection::General::HS::HSOrientation GSQ_Orientation(double a,double b,double g)
    {
        RunSection::General::HS::HSOrientation o;
        o.alpha=a; o.beta=b; o.gamma=g; o.weight=1.0;
        SpinAPI::CreateZYZRotationMatrix(a,b,g,o.frameToLab);
        return o;
    }

    bool GSQ_Gap(const SpinAPI::system_ptr &system,
                 const RunSection::General::HS::HSExecutionPlan &plan,
                 const RunSection::General::HS::HSOrientation &orientation,
                 double &gap, arma::cx_mat *matrix=nullptr)
    {
        if (system == nullptr) return false;
        SpinAPI::SpinSpace space(*system);
        space.UseSuperoperatorSpace(false);
        RunSection::General::HS::HSHamiltonianBuilder builder(plan,space);
        arma::sp_cx_mat H;
        std::string error;
        if (!builder.BuildStatic(orientation,H,nullptr,error)) return false;
        arma::vec eig;
        if (!arma::eig_sym(eig,arma::cx_mat(H)) || eig.n_elem < 2) return false;
        gap = eig.max()-eig.min();
        if (matrix != nullptr) *matrix = arma::cx_mat(H);
        return std::isfinite(gap);
    }

    bool GSQ_ExtractColumns(const std::string &data,const std::string &fieldName,
                            const std::string &signalName,
                            std::vector<double> &fields,std::vector<double> &signal)
    {
        std::istringstream in(data);
        std::string line;
        if (!std::getline(in,line)) return false;
        std::vector<std::string> headers;
        {
            std::istringstream hs(line);
            for (std::string token; hs >> token;) headers.push_back(token);
        }
        const auto fit=std::find(headers.begin(),headers.end(),fieldName);
        const auto sit=std::find(headers.begin(),headers.end(),signalName);
        if (fit==headers.end() || sit==headers.end()) return false;
        const size_t fi=static_cast<size_t>(std::distance(headers.begin(),fit));
        const size_t si=static_cast<size_t>(std::distance(headers.begin(),sit));
        fields.clear(); signal.clear();
        while (std::getline(in,line))
        {
            if (line.empty()) continue;
            std::istringstream ls(line);
            std::vector<double> values;
            for (std::string token; ls >> token;)
            {
                try { values.push_back(std::stod(token)); }
                catch (...) { return false; }
            }
            if (values.size() <= std::max(fi,si)) return false;
            fields.push_back(values[fi]);
            signal.push_back(values[si]);
        }
        return !fields.empty() && fields.size()==signal.size();
    }

    bool GSQ_RunLegacyIsotropicSweep(double frequencyGHz,double g,double &peakField_mT)
    {
        const double omega=2.0*arma::datum::pi*frequencyGHz;
        const double bres=omega/(GSQ_MU_B_OVER_HBAR*g);
        const double start=bres-0.004;
        const double step=0.0001;
        const int nsteps=81;

        auto system=GSQ_BuildOneElectron(g,g,g,start);
        if (system==nullptr) return false;

        RunSection::RunSection rs;
        rs.Add(system);
        MSDParser::ObjectParser settings("general","steps="+std::to_string(nsteps)+";");
        rs.Add(MSDParser::ObjectType::Settings,settings);

        std::ostringstream taskProps;
        taskProps << std::setprecision(17)
                  << "type=statichs-resonance-spectra;mwfrequency=" << frequencyGHz
                  << ";linewidth=0.2;lineshape=gaussian;detectspins=E;fieldinteraction=B0;"
                     "hamiltonianh0list=B0;powdersamplingpoints=1;powdergridtype=uniform;"
                     "powdergammapoints=1;powderfullsphere=true;fulltensorrotation=true;"
                     "sweepcache=false;initialstate=Up;";
        MSDParser::ObjectParser taskParser("res",taskProps.str());
        if (!rs.Add(MSDParser::ObjectType::Task,taskParser)) return false;
        auto task=rs.GetTask("res");
        if (task==nullptr) return false;

        MSDParser::ObjectParser action("field",
            "type=addvector;vector=System.B0.field;direction=0 0 1;value="+std::to_string(step)+";");
        if (!rs.Add(MSDParser::ObjectType::Action,action)) return false;

        std::ostringstream log,data;
        task->SetLogStream(log); task->SetDataStream(data);
        for (int s=1;s<=nsteps;++s)
        {
            if (!rs.Run(static_cast<unsigned int>(s))) return false;
            if (s<nsteps && !rs.Step(static_cast<unsigned int>(s+1))) return false;
        }

        std::vector<double> fields,signal;
        if (!GSQ_ExtractColumns(data.str(),"System.Field_mT","System.Total_perp",fields,signal)) return false;
        size_t best=0;
        double bestAbs=-1.0;
        for (size_t i=0;i<signal.size();++i)
        {
            const double a=std::abs(signal[i]);
            if (a>bestAbs) { bestAbs=a; best=i; }
        }
        if (!(bestAbs>0.0) || best>=fields.size()) return false;
        peakField_mT=fields[best];
        return std::isfinite(peakField_mT);
    }

    bool GSQ_TestIsotropicAnalyticAndOrientationInvariant()
    {
        const double fGHz=9.5;
        const double g=2.0023;
        const double omega=2.0*arma::datum::pi*fGHz;
        const double bres=omega/(GSQ_MU_B_OVER_HBAR*g);
        auto system=GSQ_BuildOneElectron(g,g,g,bres);
        if (system==nullptr) return false;
        const auto plan=GSQ_MakePlan(false);
        const auto identity=GSQ_Orientation(0.0,0.0,0.0);
        const auto rotated=GSQ_Orientation(0.73,1.11,-0.42);
        double gap0=0.0,gap1=0.0;
        arma::cx_mat H0,H1;
        if (!GSQ_Gap(system,plan,identity,gap0,&H0) || !GSQ_Gap(system,plan,rotated,gap1,&H1)) return false;
        const double tol=2e-11*std::max(1.0,omega);
        return std::abs(gap0-omega)<=tol && std::abs(gap1-omega)<=tol &&
               arma::norm(H0-H1,"fro")<=tol;
    }

    bool GSQ_TestAxialPrincipalAxisEdges()
    {
        const double fGHz=9.5;
        const double omega=2.0*arma::datum::pi*fGHz;
        const double gPerp=2.0;
        const double gPar=2.4;
        const double bPar=omega/(GSQ_MU_B_OVER_HBAR*gPar);
        const double bPerp=omega/(GSQ_MU_B_OVER_HBAR*gPerp);
        if (!(bPar < bPerp)) return false;

        const auto full=GSQ_MakePlan(false);
        const auto secular=GSQ_MakePlan(true);
        double gf=0.0,gs=0.0;
        auto sysPar=GSQ_BuildOneElectron(gPerp,gPerp,gPar,bPar);
        if (!GSQ_Gap(sysPar,full,GSQ_Orientation(0,0,0),gf) ||
            !GSQ_Gap(sysPar,secular,GSQ_Orientation(0,0,0),gs)) return false;
        const double tol=2e-10*std::max(1.0,omega);
        if (std::abs(gf-omega)>tol || std::abs(gs-omega)>tol) return false;

        auto sysPerp=GSQ_BuildOneElectron(gPerp,gPerp,gPar,bPerp);
        if (!GSQ_Gap(sysPerp,full,GSQ_Orientation(0,arma::datum::pi/2.0,0),gf) ||
            !GSQ_Gap(sysPerp,secular,GSQ_Orientation(0,arma::datum::pi/2.0,0),gs)) return false;
        return std::abs(gf-omega)<=tol && std::abs(gs-omega)<=tol;
    }

    bool GSQ_TestFullVsSecularAreDistinctConcepts()
    {
        auto system=GSQ_BuildOneElectron(2.0,2.1,2.4,0.34);
        if (system==nullptr) return false;
        const auto o=GSQ_Orientation(0.31,0.67,-0.28);
        double fullGap=0.0,secularGap=0.0;
        arma::cx_mat Hfull,Hsec;
        if (!GSQ_Gap(system,GSQ_MakePlan(false),o,fullGap,&Hfull) ||
            !GSQ_Gap(system,GSQ_MakePlan(true),o,secularGap,&Hsec)) return false;
        if (!(fullGap>0.0) || !(secularGap>0.0)) return false;
        // The full anisotropic Zeeman Hamiltonian contains transverse matrix
        // elements in the lab-z basis; the high-field/secular Hamiltonian does not.
        if (std::abs(Hfull(0,1)) < 1e-6) return false;
        if (std::abs(Hsec(0,1)) > 1e-11) return false;
        // The secular Zeeman builder preserves the first-order splitting magnitude
        // while projecting the effective field onto Sz. Thus the energy gap may
        // agree even though the operator is not the full Hamiltonian.
        if (std::abs(fullGap-secularGap) > 2e-10*std::max(1.0,fullGap)) return false;
        return arma::norm(Hfull-Hsec,"fro") > 1e-5;
    }

    bool GSQ_TestGAndHyperfineCoRotate()
    {
        arma::mat G={{2.03,0.07,-0.04},{0.07,2.21,0.05},{-0.04,0.05,2.47}};
        arma::mat A={{0.012,0.003,-0.002},{0.003,-0.006,0.004},{-0.002,0.004,0.019}};
        auto e=std::make_shared<SpinAPI::Spin>("E","type=electron;spin=1/2;tensor=matrix("+GSQ_MatrixString(G)+");");
        auto n=std::make_shared<SpinAPI::Spin>("N","type=nucleus;spin=1/2;tensor=isotropic(1);");
        auto z=std::make_shared<SpinAPI::Interaction>("B0",
            "type=zeeman;spins=E;field=0 0 0.031;ignoretensors=false;commonprefactor=false;prefactor=1.0;");
        auto hfc=std::make_shared<SpinAPI::Interaction>("A",
            "type=hyperfine;group1=E;group2=N;tensor=matrix("+GSQ_MatrixString(A)+");"
            "ignoretensors=true;commonprefactor=false;prefactor=1.0;");
        auto system=std::make_shared<SpinAPI::SpinSystem>("System");
        system->Add(e);system->Add(n);system->Add(z);system->Add(hfc);
        if (!system->ValidateInteractions().empty()) return false;

        RunSection::General::HS::HSExecutionPlan plan=GSQ_MakePlan(false);
        plan.h0List={"B0","A"};
        const auto o=GSQ_Orientation(0.43,0.81,-0.37);
        SpinAPI::SpinSpace space(*system);
        RunSection::General::HS::HSHamiltonianBuilder builder(plan,space);
        arma::sp_cx_mat Hsp;
        std::string error;
        if (!builder.BuildStatic(o,Hsp,nullptr,error)) return false;

        arma::sp_cx_mat Sx,Sy,Sz,Ix,Iy,Iz;
        if (!space.CreateOperator(e->Sx(),e,Sx) || !space.CreateOperator(e->Sy(),e,Sy) || !space.CreateOperator(e->Sz(),e,Sz) ||
            !space.CreateOperator(n->Sx(),n,Ix) || !space.CreateOperator(n->Sy(),n,Iy) || !space.CreateOperator(n->Sz(),n,Iz)) return false;
        const arma::mat Gr=o.frameToLab*G*o.frameToLab.t();
        const arma::mat Ar=o.frameToLab*A*o.frameToLab.t();
        const double Bz=0.031;
        arma::sp_cx_mat manual=Sx*(Bz*Gr(0,2))+Sy*(Bz*Gr(1,2))+Sz*(Bz*Gr(2,2));
        const arma::sp_cx_mat Sop[3]={Sx,Sy,Sz};
        const arma::sp_cx_mat Iop[3]={Ix,Iy,Iz};
        for (int a=0;a<3;++a)
            for (int b=0;b<3;++b)
                manual += Sop[a]*Iop[b]*Ar(a,b);
        const double scale=std::max({1.0,arma::norm(arma::cx_mat(Hsp),"fro"),arma::norm(arma::cx_mat(manual),"fro")});
        const double diff=arma::norm(arma::cx_mat(Hsp-manual),"fro");
        return diff <= 5e-11*scale;
    }

    bool GSQ_TestPowderSecondMomentNormalization()
    {
        RunSection::General::GeneralOrientationRequest req;
        req.mode=RunSection::General::GeneralOrientationMode::Powder2D;
        req.gridType=SpinAPI::PowderGridType::Uniform;
        req.domain=SpinAPI::PowderGridDomain::FullSphere;
        req.powderPoints=1200;
        req.owner="General spectroscopy qualification";
        std::vector<RunSection::General::GeneralOrientation> orientations;
        std::ostringstream log;
        std::string error;
        if (!RunSection::General::GeneralOrientationSampler::Build(req,orientations,log,error)) return false;
        arma::mat G=arma::diagmat(arma::vec({2.0,2.1,2.4}));
        double wsum=0.0,avg=0.0;
        for (const auto &o:orientations)
        {
            const arma::mat Gl=o.frameToLab*G*o.frameToLab.t();
            const arma::vec v=Gl.col(2);
            wsum += o.weight;
            avg += o.weight*arma::dot(v,v);
        }
        const double analytic=(4.0+4.41+5.76)/3.0;
        return std::abs(wsum-1.0)<1e-12 && std::abs(avg-analytic)<2.5e-3;
    }

    bool GSQ_TestFrozenLegacyIsotropicPeak()
    {
        const double fGHz=9.5;
        const double g=2.0023;
        const double expected=1.0e3*(2.0*arma::datum::pi*fGHz)/(GSQ_MU_B_OVER_HBAR*g);
        double peak=0.0;
        if (!GSQ_RunLegacyIsotropicSweep(fGHz,g,peak)) return false;
        // 0.1 mT field spacing; allow one grid interval for the sampled maximum.
        return std::abs(peak-expected)<=0.11;
    }
}

void AddGeneralSpectroscopyQualificationTests(std::vector<test_case> &cases)
{
    cases.push_back(test_case("General spectroscopy isotropic-g analytic resonance and orientation invariance",GSQ_TestIsotropicAnalyticAndOrientationInvariant));
    cases.push_back(test_case("General spectroscopy axial-g principal-axis resonance edges",GSQ_TestAxialPrincipalAxisEdges));
    cases.push_back(test_case("General spectroscopy high-field secularization is distinct from full Hamiltonian",GSQ_TestFullVsSecularAreDistinctConcepts));
    cases.push_back(test_case("General spectroscopy anisotropic g and hyperfine co-rotate",GSQ_TestGAndHyperfineCoRotate));
    cases.push_back(test_case("General spectroscopy powder second moment and normalized weights",GSQ_TestPowderSecondMomentNormalization));
    cases.push_back(test_case("General spectroscopy frozen legacy isotropic resonance peak",GSQ_TestFrozenLegacyIsotropicPeak));
}
