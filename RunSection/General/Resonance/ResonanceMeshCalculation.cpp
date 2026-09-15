/////////////////////////////////////////////////////////////////////////
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "ResonanceCalculations.h"
#include "ResonanceOrientationSampler.h"
#include "ResonanceSystemPreparation.h"
#include "ExactResonanceSolver.h"
#include "ResonancePowderMesh.h"
#include "SpinSystem.h"
#include "Interaction.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>

namespace RunSection::General::Resonance
{
    bool ResonanceMeshCalculation::Sweep(const ResonanceExecutionPlan &plan,
                                         const SpinAPI::system_ptr &system,
                                         const SpinAPI::interaction_ptr &field,
                                         const ResonanceFieldSweep &sweep, ResonanceSpectrum &cache,
                                         std::ostream &log, std::string &error)
    {
        error.clear();
        const auto &field0 = sweep.field0;
        const auto &fieldStep = sweep.fieldStep;

        const unsigned steps = sweep.steps;
        if (!system || !field || steps < 2 || field0.n_elem != 3 || fieldStep.n_elem != 3 ||
            !(field0(2) > 0) || !(fieldStep(2) > 0) ||
            std::abs(field0(0)) + std::abs(field0(1)) + std::abs(fieldStep(0)) + std::abs(fieldStep(1)) >
                1e-12 ||
            !plan.fullTensorRotation || !plan.powderFullSphere || plan.powderGammaPoints != 1 ||
            system->InitialStateFrame() != SpinAPI::StateFrame::Eigen || !plan.initialStateName.empty())
        {
            log << "powdermesh currently requires full-sphere/full-tensor, gamma=1, a positive ascending z "
                   "field "
                   "and full-Hamiltonian thermal equilibrium."
                << std::endl;
            return false;
        }
        auto states = system->InitialState();
        if (states.size() != 1 || states.front() != nullptr || !(system->Temperature() > 0))
            return false;
        std::vector<std::string> hnames = plan.hamiltonianH0list;
        if (hnames.empty())
            for (const auto &interaction : system->Interactions())
                if (SpinAPI::IsStatic(*interaction))
                    hnames.push_back(interaction->Name());
        auto thermal = system->ThermalHamiltonianList();
        auto ordered = hnames;
        std::sort(ordered.begin(), ordered.end());
        std::sort(thermal.begin(), thermal.end());
        if (ordered != thermal)
        {
            log << "powdermesh requires thermalhamiltonian to equal hamiltonianh0list." << std::endl;
            return false;
        }
        auto zeeman = SystemPreparation::CollectZeemanInteractions(system, hnames);
        std::vector<std::string> statics, zeenames;
        for (const auto &name : hnames)
        {
            bool isField = false;
            for (const auto &z : zeeman)
                if (z->Name() == name)
                    isField = true;
            (isField ? zeenames : statics).push_back(name);
        }
        std::vector<SpinAPI::spin_ptr> spins;
        std::vector<std::string> names;
        if (!SystemPreparation::ResolveDetectionSpins(plan, system, field, spins, names) || spins.empty())
            return false;
        std::vector<ResonanceMagneticMomentTerm> terms;
        if (!SystemPreparation::PrepareMoments(spins, zeeman, terms, error))
            return false;
        SpinAPI::SpinSpace space(*system);
        space.UseSuperoperatorSpace(false);
        space.UseFullTensorRotation(true);
        struct Prepared
        {
            arma::cx_mat h0, dhdB, mx, my;
            arma::sp_cx_mat derivative;
            std::vector<ResonanceDetectionOperator> channels;
        };
        const int nu = plan.meshCosPoints, np = plan.meshPhiPoints, nf = plan.meshFieldPoints;
        const int count = nu * np;
        std::vector<Prepared> prepared(count);
        const bool clusterAxes = plan.meshClusterAxes;
        if (clusterAxes && ((nu - 1) % 2 != 0 || np % 4 != 0))
        {
            log << "Axis-clustered mesh requires odd meshcospoints and meshphipoints divisible by four."
                << std::endl;
            return false;
        }
        std::vector<double> u(nu), azimuth(np + 1);
        auto angle = [&](int index, int intervals, int quadrants)
        {
            if (!clusterAxes)
                return quadrants * arma::datum::pi * .5 * index / intervals;
            int perQuadrant = intervals / quadrants, sector = std::min(index / perQuadrant, quadrants - 1);
            double t = double(index - sector * perQuadrant) / perQuadrant;
            return arma::datum::pi * .5 * (sector + .5 * (1 - std::cos(arma::datum::pi * t)));
        };
        for (int i = 0; i < nu; ++i)
            u[i] = -std::cos(angle(i, nu - 1, 2));
        for (int j = 0; j <= np; ++j)
            azimuth[j] = angle(j, np, 4);
        for (int i = 0; i < nu; ++i)
            for (int j = 0; j < np; ++j)
            {
                double theta = std::acos(std::clamp(u[i], -1., 1.)), phi = azimuth[j], gamma = 0.;
                arma::mat rotation;
                OrientationSampling::Rotation(phi, theta, gamma, rotation);
                auto &p = prepared[i * np + j];
                arma::sp_cx_mat hs, hz;
                if (statics.empty())
                    hs = arma::sp_cx_mat(space.HilbertSpaceDimensions(), space.HilbertSpaceDimensions());
                else if (!SystemPreparation::BuildHamiltonian(space, statics, rotation, hs, error))
                    return false;
                if (!SystemPreparation::BuildHamiltonian(space, zeenames, rotation, hz, error) ||
                    !ResonanceMagneticMomentBuilder::BuildTransverseChannels(space, terms, rotation, true,
                                                                             p.channels, error))
                    return false;
                p.h0 = arma::cx_mat(hs);
                p.derivative = hz / field0(2);
                p.dhdB = arma::cx_mat(p.derivative);
                p.mx = arma::zeros<arma::cx_mat>(p.h0.n_rows, p.h0.n_cols);
                p.my = p.mx;
                for (const auto &channel : p.channels)
                {
                    p.mx += channel.x;
                    p.my += channel.y;
                }
            }
        const double db = fieldStep(2), width = plan.linewidth_mT;
        // Preserve enough source range for later linewidth checks up to 2*width.
        const int padding = static_cast<int>(std::ceil(12e-3 * width / db)) + 2;
        const double first = field0(2) - padding * db;
        const int bins = steps + 2 * padding;
        const double blo = std::max(1e-9, first - .5 * db), bhi = first + (bins - .5) * db;
        std::vector<double> meshFields(nf);
        for (int i = 0; i < nf; ++i)
        {
            double t = double(i) / (nf - 1), scale = plan.meshFieldScale;
            meshFields[i] =
                scale > 0 ? scale * std::sinh((1 - t) * std::asinh(blo / scale) + t * std::asinh(bhi / scale))
                          : (1 - t) * blo + t * bhi;
        }
        const size_t channels = 5 + 5 * spins.size();
        std::vector<std::vector<double>> mass(channels, std::vector<double>(bins, 0.));
        ResonancePowderMesh mesh(first, db, mass);
        std::vector<ResonanceMeshNode> previous(count), current(count);
        log << "Powder resonance surface mesh: " << nu << " x " << np << " orientations, " << nf
            << " field planes; exact full thermal Hamiltonian." << std::endl;
        for (int f = 0; f < nf; ++f)
        {
            std::vector<int> success(count, 1);
#pragma omp parallel for schedule(static)
            for (int v = 0; v < count; ++v)
            {
                const auto &p = prepared[v];
                auto &node = current[v];
                ResonanceLineSet lines;
                std::string error;
                if (!ExactResonanceSolver::GenerateFrequencyThermal(p.h0 + meshFields[f] * p.dhdB,
                                                                    system->Temperature(), p.derivative, p.mx,
                                                                    p.my, lines, error, p.channels))
                {
                    success[v] = 0;
                    continue;
                }
                node.omega.resize(lines.lines.size());
                node.strength.resize(lines.lines.size());
                for (size_t l = 0; l < lines.lines.size(); ++l)
                {
                    const auto &line = lines.lines[l];
                    const auto &m = line.moment;
                    node.omega[l] = line.omega;
                    auto &w = node.strength[l];
                    w.resize(channels);
                    w[0] = m.x;
                    w[1] = m.y;
                    w[2] = m.perpendicular;
                    w[3] = m.crossX;
                    w[4] = m.crossY;
                    for (size_t k = 0; k < spins.size(); ++k)
                    {
                        const auto &s = m.channels[k];
                        size_t o = 5 + 5 * k;
                        w[o] = s.x;
                        w[o + 1] = s.y;
                        w[o + 2] = s.perpendicular;
                        w[o + 3] = s.plus;
                        w[o + 4] = s.minus;
                    }
                    for (auto &value : w)
                        value *= line.populationDifference;
                }
            }
            if (std::find(success.begin(), success.end(), 0) != success.end())
            {
                log << "Exact frequency-surface preparation failed." << std::endl;
                return false;
            }
            if (f > 0)
                for (int i = 0; i < nu - 1; ++i)
                    for (int j = 0; j < np; ++j)
                    {
                        int next = (j + 1) % np;
                        std::array<const ResonanceMeshNode *, 8> cell = {
                            &previous[i * np + j],    &previous[(i + 1) * np + j],
                            &previous[i * np + next], &previous[(i + 1) * np + next],
                            &current[i * np + j],     &current[(i + 1) * np + j],
                            &current[i * np + next],  &current[(i + 1) * np + next]};
                        mesh.AddCell(cell, meshFields[f - 1], meshFields[f], u[i + 1] - u[i],
                                     azimuth[j + 1] - azimuth[j], 2 * arma::datum::pi * plan.mwFrequencyGHz);
                    }
            previous.swap(current);
            if (f % std::max(1, nf / 10) == 0)
                log << "Powder mesh field plane " << f + 1 << "/" << nf << std::endl;
        }
        cache.steps = steps;
        cache.spin_names = names;
        cache.field_mT.resize(steps);
        for (unsigned i = 0; i < steps; ++i)
            cache.field_mT[i] = 1000 * (field0(2) + i * db);
        const auto &rawFile = plan.meshRawFile;
        if (!rawFile.empty())
        {
            std::ofstream raw(rawFile);
            if (!raw)
            {
                log << "Could not write powder-mesh bin masses." << std::endl;
                return false;
            }
            raw << "field_mT,integrated_transverse_line_weight\n" << std::setprecision(17);
            for (int i = 0; i < bins; ++i)
                raw << 1000 * (first + i * db) << "," << mass[2][i] << "\n";
        }
        // Zero-padded FFT implements the same normalized field Gaussian as the
        // root route; bin masses already include the frequency delta integral.
        size_t fftSize = 1;
        while (fftSize < static_cast<size_t>(bins + 2 * padding))
            fftSize *= 2;
        arma::vec kernel(fftSize, arma::fill::zeros);
        for (int k = -padding; k <= padding; ++k)
            kernel[(k + fftSize) % fftSize] =
                ResonanceSpectrumProcessing::LineshapeValue(plan, 1000 * k * db, width);
        const arma::cx_vec kernelFFT = arma::fft(kernel);
        auto broaden = [&](size_t channel, std::vector<double> &out)
        {
            arma::vec source(fftSize, arma::fill::zeros);
            for (int j = 0; j < bins; ++j)
                source[j] = mass[channel][j];
            arma::vec broadened = arma::real(arma::ifft(arma::fft(source) % kernelFFT));
            out.resize(steps);
            for (unsigned j = 0; j < steps; ++j)
                out[j] = broadened[j + padding];
        };
        broaden(0, cache.total_x);
        broaden(1, cache.total_y);
        broaden(2, cache.total_perp);
        broaden(3, cache.cross_x);
        broaden(4, cache.cross_y);
        cache.spin_x.resize(spins.size());
        cache.spin_y.resize(spins.size());
        cache.spin_perp.resize(spins.size());
        cache.spin_p.resize(spins.size());
        cache.spin_m.resize(spins.size());
        for (size_t k = 0; k < spins.size(); ++k)
        {
            broaden(5 + 5 * k, cache.spin_x[k]);
            broaden(6 + 5 * k, cache.spin_y[k]);
            broaden(7 + 5 * k, cache.spin_perp[k]);
            broaden(8 + 5 * k, cache.spin_p[k]);
            broaden(9 + 5 * k, cache.spin_m[k]);
        }
        for (double value : cache.total_perp)
            if (!std::isfinite(value))
                return false;
        ResonanceSpectrumProcessing::ApplyDetectionHarmonic(plan, cache);
        return true;
    }
} // namespace RunSection::General::Resonance
