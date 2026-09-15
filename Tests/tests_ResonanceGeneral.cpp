//////////////////////////////////////////////////////////////////////////////
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
// General-task migration qualification. The frozen legacy task remains an
// independent execution reference; the existing physical fixtures are shared.
//////////////////////////////////////////////////////////////////////////////
#include "TaskResonanceGeneral.h"
#include "ResonanceSystemPreparation.h"
#include "ResonanceOrientationSampler.h"
#include <functional>
#include <iomanip>
#include <sstream>

namespace ResonanceGeneralTests
{
    using namespace RunSection::General::Resonance;
    const std::string legacy = "statichs-resonance-spectra", general = "ResonanceGeneral";
    struct Result
    {
        bool ok = false;
        std::string data, log;
    };
    Result Run(const std::vector<SpinAPI::system_ptr> &systems, const std::string &type,
               const std::string &properties, unsigned steps = 41, const std::string &field = "Z",
               double increment = .0045, bool actions = true)
    {
        RunSection::RunSection run;
        for (const auto &system : systems)
        {
            if (!system)
                return {};
            run.Add(system);
        }
        run.Add(
            MSDParser::ObjectType::Settings,
            MSDParser::ObjectParser("general", "steps=" + std::to_string(steps) + ";outputprecision=17;"));
        if (!run.Add(MSDParser::ObjectType::Task,
                     MSDParser::ObjectParser("resonance", "type=" + type + ";" + properties)))
            return {};
        const auto task = run.GetTask("resonance");
        if (!task)
            return {};
        if (actions)
            for (const auto &system : systems)
            {
                std::ostringstream p;
                p << std::setprecision(17) << "type=addvector;vector=" << system->Name() << "." << field
                  << ".field;direction=0 0 1;value=" << increment << ";";
                if (!run.Add(MSDParser::ObjectType::Action,
                             MSDParser::ObjectParser(system->Name() + "sweep", p.str())))
                    return {};
            }
        std::ostringstream data, log;
        data << std::setprecision(17);
        task->SetDataStream(data);
        task->SetLogStream(log);
        bool ok = true;
        for (unsigned step = 1; step <= steps; ++step)
        {
            if (!run.Run(step))
            {
                ok = false;
                break;
            }
            if (step < steps && !run.Step(step + 1))
            {
                ok = false;
                break;
            }
        }
        return {ok, data.str(), log.str()};
    }
    bool Table(const std::string &data, std::vector<std::string> &header,
               std::vector<std::vector<double>> &rows)
    {
        std::istringstream in(data);
        std::string line;
        header.clear();
        rows.clear();
        if (!std::getline(in, line))
            return false;
        std::istringstream h(line);
        for (std::string word; h >> word;)
            header.push_back(word);
        while (std::getline(in, line))
        {
            if (line.empty())
                continue;
            std::vector<double> row;
            std::istringstream values(line);
            for (double v; values >> v;)
            {
                if (!std::isfinite(v))
                    return false;
                row.push_back(v);
            }
            if (row.size() != header.size())
                return false;
            rows.push_back(std::move(row));
        }
        return !rows.empty();
    }
    bool Equal(const Result &a, const Result &b, double tolerance = 2e-9)
    {
        if (!a.ok || !b.ok)
        {
            std::cerr << a.log << b.log;
            return false;
        }
        std::vector<std::string> ha, hb;
        std::vector<std::vector<double>> ra, rb;
        if (!Table(a.data, ha, ra) || !Table(b.data, hb, rb) || ha != hb || ra.size() != rb.size())
            return false;
        double peak = 0, delta = 0;
        for (size_t r = 0; r < ra.size(); ++r)
            for (size_t c = 0; c < ha.size(); ++c)
            {
                if (c < 3)
                {
                    if (std::abs(ra[r][c] - rb[r][c]) > 1e-8)
                        return false;
                    continue;
                }
                peak = std::max(peak, std::max(std::abs(ra[r][c]), std::abs(rb[r][c])));
                delta = std::max(delta, std::abs(ra[r][c] - rb[r][c]));
            }
        if (!(peak > 1e-15) || delta > tolerance * peak + 1e-13)
        {
            std::cerr << "resonance parity: peak=" << peak << " delta=" << delta << std::endl;
            return false;
        }
        return true;
    }
    std::string TripletOptions(const std::string &cache, const std::string &grid = "fibonacci", int gamma = 1,
                               int harmonic = 0, double width = 2, const std::string &shape = "gaussian")
    {
        return "mwfrequency=9.75;linewidth=" + std::to_string(width) + ";lineshape=" + shape +
               ";detectspins=E;fieldinteraction=Z;hamiltonianh0list=D,Z;powdergridtype=" + grid +
               ";powdersamplingpoints=12;powdergridsize=7;powdergridsymmetry=C1;powdergammapoints=" +
               std::to_string(gamma) + ";powderfullsphere=true;fulltensorrotation=true;mzblocks=true;" +
               "harmonic=" + std::to_string(harmonic) + ";modamp=1;" + cache;
    }
    bool TripletParity(const std::string &cache, const std::string &thermal = "",
                       const std::string &grid = "fibonacci", int gamma = 1, int harmonic = 0,
                       double width = 2, const std::string &shape = "gaussian")
    {
        const auto options = TripletOptions(cache, grid, gamma, harmonic, width, shape);
        return Equal(Run({BuildComplexTripletSystem(304.2, thermal)}, legacy, options),
                     Run({BuildComplexTripletSystem(304.2, thermal)}, general, options));
    }
    bool PointMolecular()
    {
        return TripletParity("sweepcache=false;");
    }
    bool ExactMolecular()
    {
        return TripletParity("sweepcache=true;");
    }
    bool ThermalFull()
    {
        return TripletParity("sweepcache=true;", "D,Z");
    }
    bool ThermalSubset()
    {
        return TripletParity("sweepcache=true;", "D");
    }
    bool PointThermal()
    {
        return TripletParity("sweepcache=false;", "D,Z");
    }
    bool SopheGamma()
    {
        return TripletParity("sweepcache=true;", "", "sophe", 3);
    }
    bool Lorentzian()
    {
        return TripletParity("sweepcache=true;", "D,Z", "fibonacci", 1, 0, 2, "lorentzian");
    }
    bool FirstHarmonic()
    {
        return TripletParity("sweepcache=true;", "D,Z", "fibonacci", 1, 1, 6);
    }
    bool SecondHarmonic()
    {
        return TripletParity("sweepcache=true;", "", "fibonacci", 1, 2, 6);
    }
    bool Approximate()
    {
        return TripletParity("sweepcache=true;sweepcachemode=approx;");
    }
    bool Projection()
    {
        return TripletParity("sweepcache=true;sweepcachemode=resonanceprojection;resfieldspoints=21;");
    }
    bool ZeroWidthProjection()
    {
        return TripletParity("sweepcache=true;sweepcachemode=resonanceprojection;resfieldspoints=41;", "",
                             "sophe", 1, 0, 0);
    }
    bool RefinedRoots()
    {
        return TripletParity("sweepcache=true;sweepcachemode=refinedroots;", "D,Z");
    }
    bool PowderMesh()
    {
        return TripletParity("sweepcache=true;sweepcachemode=powdermesh;meshcospoints=7;meshphipoints=12;"
                             "meshfieldpoints=21;meshclusteraxes=true;",
                             "D,Z");
    }
    bool ExactAgainstPoint()
    {
        return Equal(
            Run({BuildComplexTripletSystem(304.2, "D,Z")}, general, TripletOptions("sweepcache=true;")),
            Run({BuildComplexTripletSystem(304.2, "D,Z")}, general, TripletOptions("sweepcache=false;")));
    }
    bool TwoZeemanParity()
    {
        const std::string props =
            "mwfrequency=94;linewidth=2;lineshape=gaussian;detectspins=FE1,WE2;"
            "fieldinteraction=zeeman1;hamiltonianh0list=zeeman1,zeeman2;initialstate=Init;"
            "powdergridtype=fibonacci;powdersamplingpoints=12;powdergammapoints=2;"
            "fulltensorrotation=false;enforce_zeeman_sync=true;sweepcache=true;";
        auto a = BuildTwoZeemanSystem(3.33, 3.10), b = BuildTwoZeemanSystem(3.33, 3.10);
        if (!a->states_find("Init")->ParseFromSystem(*a) || !b->states_find("Init")->ParseFromSystem(*b))
            return false;
        const auto ra = Run({a}, legacy, props, 21, "zeeman1", .003),
                   rb = Run({b}, general, props, 21, "zeeman1", .003);
        // Temporary sync must not change the other interaction's physical field.
        return std::abs(b->interactions_find("zeeman2")->Field()(2) - 3.10) < 1e-14 && Equal(ra, rb);
    }
    bool HybridParity(bool thermal)
    {
        auto build = [&]()
        {
            return thermal ? BuildR2KCThermalOneVSystem(.336, .0047, "B0") : BuildR2KBOneVSystem(.336, false);
        };
        const std::string props =
            "mwfrequency=9.5;linewidth=.2;lineshape=gaussian;detectspins=E;"
            "fieldinteraction=B0;hamiltonianh0list=B0,A,NZ;powdergridtype=sophe;powdergridsize=5;"
            "powdergridsymmetry=Dinfh;powdergammapoints=1;fulltensorrotation=true;"
            "enforce_zeeman_sync=true;sweepcache=false;solver=hybrid;perturbativenuclei=V;";
        return Equal(Run({build()}, legacy, props, 9, "B0", .0008),
                     Run({build()}, general, props, 9, "B0", .0008));
    }
    bool HybridFixed()
    {
        return HybridParity(false);
    }
    bool HybridThermal()
    {
        return HybridParity(true);
    }
    SpinAPI::system_ptr Isotropic(const std::string &name, double field = 0.31)
    {
        auto system = std::make_shared<SpinAPI::SpinSystem>(name);
        system->Add(std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1/2;tensor=isotropic(2);"));
        system->Add(std::make_shared<SpinAPI::Interaction>(
            "Z", "type=zeeman;spins=E;field=0 0 " + std::to_string(field) +
                     ";commonprefactor=true;ignoretensors=false;"));
        auto down = std::make_shared<SpinAPI::State>("Down", "spin(E)=|-1/2>;");
        system->Add(down);
        system->ValidateInteractions();
        down->ParseFromSystem(*system);
        system->SetProperties(
            std::make_shared<MSDParser::ObjectParser>("population", "initialstate=Down;frame=fixed;"));
        return system;
    }
    const std::string isoOptions = "mwfrequency=9.5;linewidth=1;detectspins=E;fieldinteraction=Z;"
                                   "powdergridtype=fibonacci;powdersamplingpoints=1;sweepcache=true;";
    bool AnalyticField()
    {
        const auto result = Run({Isotropic("Iso")}, general, isoOptions, 201, "Z", .00025);
        std::vector<double> b, y;
        if (!result.ok || !ExtractColumn(result.data, "Iso.Field_mT", b) ||
            !ExtractColumn(result.data, "Iso.Total_perp", y))
            return false;
        const size_t maximum = std::distance(y.begin(), std::max_element(y.begin(), y.end()));
        // Independent g=2 X-band position and Gaussian absorption FWHM.
        const double expected = 1000 * 9.5 / (2 * 13.99624555);
        if (std::abs(b[maximum] - expected) > .15)
            return false;
        for (size_t i = 0; i < y.size(); ++i)
        {
            const double target = std::exp(-4 * std::log(2.) * std::pow(b[i] - expected, 2));
            const double ratio = y[i] / y[maximum];
            if (std::abs(ratio - target / std::exp(-4 * std::log(2.) * std::pow(b[maximum] - expected, 2))) >
                3e-4)
                return false;
        }
        return true;
    }
    bool MultipleSystems()
    {
        const auto both = Run({Isotropic("A"), Isotropic("B", .32)}, general, isoOptions, 81, "Z", .0005);
        const auto one = Run({Isotropic("A")}, general, isoOptions, 81, "Z", .0005);
        std::vector<std::string> h;
        std::vector<std::vector<double>> rows;
        std::vector<double> a, b;
        return both.ok && one.ok && Table(both.data, h, rows) && rows.size() == 81 && h.size() == 24 &&
               ExtractColumn(both.data, "A.Total_perp", a) && ExtractColumn(one.data, "A.Total_perp", b) &&
               a == b;
    }
    bool InvalidPlans()
    {
        for (const auto &option :
             {"mwfrequency=0;", "mwfrequency=9.5;sweepcachemode=typo;", "mwfrequency=9.5;lineshape=typo;",
              "mwfrequency=9.5;solver=hybrid;sweepcache=false;", "mwfrequency=9.5;solver=auto;",
              "mwfrequency=9.5;solver=hybrid;perturbativenuclei=V;"})
        {
            ResonanceExecutionPlan p;
            std::string error;
            std::ostringstream log;
            if (ResolveResonanceExecutionPlan(MSDParser::ObjectParser("bad", option), p, log, error))
                return false;
        }
        return true;
    }
    bool MissingSweepRejected()
    {
        const auto result =
            Run({BuildComplexTripletSystem(304.2, "D,Z")}, general,
                TripletOptions("sweepcache=true;sweepcachemode=refinedroots;"), 1, "Z", .001, false);
        return !result.ok && result.log.find("needs a valid sweep") != std::string::npos;
    }
    bool DerivativeSweepRejected()
    {
        const auto result = Run({Isotropic("Iso")}, general, isoOptions + "harmonic=1;", 2, "Z", .001);
        return !result.ok && result.log.find("three distinct") != std::string::npos;
    }
    bool DescendingSweep()
    {
        const auto props = TripletOptions("sweepcache=true;", "fibonacci", 1, 1, 6);
        auto a = BuildComplexTripletSystem(304.2, "D,Z"), b = BuildComplexTripletSystem(304.2, "D,Z");
        arma::vec field = {0, 0, .43};
        a->interactions_find("Z")->SetField(field);
        b->interactions_find("Z")->SetField(field);
        return Equal(Run({a}, legacy, props, 41, "Z", -.0045), Run({b}, general, props, 41, "Z", -.0045));
    }
    bool AliasAndDefaults()
    {
        const std::string props = "frequency=9.5;linewidth=1;";
        return Equal(Run({Isotropic("Iso", .339)}, legacy, props, 1, "Z", .001, false),
                     Run({Isotropic("Iso", .339)}, "resonance-general", props, 1, "Z", .001, false));
    }
    bool RebuildOnRestart()
    {
        auto system = Isotropic("Iso");
        RunSection::RunSection run;
        run.Add(system);
        run.Add(MSDParser::ObjectType::Settings, MSDParser::ObjectParser("general", "steps=81;"));
        run.Add(MSDParser::ObjectType::Task,
                MSDParser::ObjectParser("resonance", "type=ResonanceGeneral;" + isoOptions));
        run.Add(MSDParser::ObjectType::Action,
                MSDParser::ObjectParser("sweep",
                                        "type=addvector;vector=Iso.Z.field;direction=0 0 1;value=.0005;"));
        auto task = run.GetTask("resonance");
        if (!task)
            return false;
        std::ostringstream first, second, log;
        task->SetLogStream(log);
        task->SetDataStream(first);
        if (!run.Run(1))
            return false;
        arma::vec field = {0, 0, .339};
        system->interactions_find("Z")->SetField(field);
        task->SetDataStream(second);
        if (!run.Run(1))
            return false;
        std::vector<double> values;
        return ExtractColumn(second.str(), "Iso.Field_mT", values) && values.size() == 1 &&
               std::abs(values[0] - 339) < 1e-10;
    }
    bool InvalidFieldRejected()
    {
        const auto result = Run({Isotropic("Iso")}, general, "mwfrequency=9.5;fieldinteraction=missing;", 1,
                                "Z", .001, false);
        return !result.ok;
    }
    bool SharedStateAndRotation()
    {
        const auto system = BuildComplexTripletSystem();
        SpinAPI::SpinSpace space(system);
        ResonanceExecutionPlan plan;
        RunSection::General::HS::HSPreparedState state;
        arma::cx_mat generalDensity, expected;
        arma::mat rotation;
        std::string error;
        if (!SystemPreparation::PrepareState(plan, system, space, state, error) ||
            !OrientationSampling::Rotation(.37, 1.13, .61, rotation) ||
            !SystemPreparation::OrientState(space, state, rotation, generalDensity, error) ||
            !space.RotateState(state.density, rotation, expected))
            return false;
        return arma::norm(generalDensity - expected, "fro") < 1e-12 &&
               arma::norm(generalDensity - state.density, "fro") > .1 &&
               std::abs(arma::trace(generalDensity) - 1.) < 1e-12;
    }
} // namespace ResonanceGeneralTests
void AddResonanceGeneralTests(std::vector<test_case> &cases)
{
    using namespace ResonanceGeneralTests;
    cases.emplace_back("ResonanceGeneral field-local molecular state legacy parity", PointMolecular);
    cases.emplace_back("ResonanceGeneral exact sweep weighted molecular state legacy parity", ExactMolecular);
    cases.emplace_back("ResonanceGeneral full Gibbs sweep legacy parity", ThermalFull);
    cases.emplace_back("ResonanceGeneral selected thermal Hamiltonian legacy parity", ThermalSubset);
    cases.emplace_back("ResonanceGeneral field-local full Gibbs legacy parity", PointThermal);
    cases.emplace_back("ResonanceGeneral SOPHE gamma and all channels legacy parity", SopheGamma);
    cases.emplace_back("ResonanceGeneral Lorentzian legacy parity", Lorentzian);
    cases.emplace_back("ResonanceGeneral first harmonic modulation legacy parity", FirstHarmonic);
    cases.emplace_back("ResonanceGeneral second harmonic modulation legacy parity", SecondHarmonic);
    cases.emplace_back("ResonanceGeneral approximate crossings legacy parity", Approximate);
    cases.emplace_back("ResonanceGeneral resonance projection legacy parity", Projection);
    cases.emplace_back("ResonanceGeneral zero-width SOPHE projection legacy parity", ZeroWidthProjection);
    cases.emplace_back("ResonanceGeneral refined roots legacy parity", RefinedRoots);
    cases.emplace_back("ResonanceGeneral experimental powder mesh legacy parity", PowderMesh);
    cases.emplace_back("ResonanceGeneral cached versus field-local full Gibbs", ExactAgainstPoint);
    cases.emplace_back("ResonanceGeneral Zeeman ownership sync restoration legacy tensor parity",
                       TwoZeemanParity);
    cases.emplace_back("ResonanceGeneral explicit hybrid fixed state legacy parity", HybridFixed);
    cases.emplace_back("ResonanceGeneral explicit hybrid core thermal legacy parity", HybridThermal);
    cases.emplace_back("ResonanceGeneral analytic g=2 X-band Gaussian", AnalyticField);
    cases.emplace_back("ResonanceGeneral independent systems output and action lifecycle", MultipleSystems);
    cases.emplace_back("ResonanceGeneral invalid numerical plans fail", InvalidPlans);
    cases.emplace_back("ResonanceGeneral required sweep cannot silently fall back", MissingSweepRejected);
    cases.emplace_back("ResonanceGeneral invalid field selection fails", InvalidFieldRejected);
    cases.emplace_back("ResonanceGeneral derivative needs three distinct fields", DerivativeSweepRejected);
    cases.emplace_back("ResonanceGeneral descending derivative sweep legacy parity", DescendingSweep);
    cases.emplace_back("ResonanceGeneral aliases and unchanged defaults", AliasAndDefaults);
    cases.emplace_back("ResonanceGeneral restart rebuilds the sweep cache", RebuildOnRestart);
    cases.emplace_back("ResonanceGeneral shared molecular state rotation", SharedStateAndRotation);
}
