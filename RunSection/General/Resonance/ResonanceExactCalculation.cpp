/////////////////////////////////////////////////////////////////////////
// Exact field-local powder response using shared General preparation/solvers.
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "ResonanceCalculations.h"
#include "ResonanceSystemPreparation.h"
#include "ResonanceOrientationSampler.h"
#include "ResonanceDiagonalization.h"
#include "ExactResonanceSolver.h"
#include "ResonanceSpectrumEvaluator.h"
#include "SpinSystem.h"
#include "Interaction.h"
#include <cmath>
#include <ostream>

namespace RunSection::General::Resonance
{
    bool ResonanceExactCalculation::Point(const ResonanceExecutionPlan &plan,
                                          const SpinAPI::system_ptr &system, ResonanceSample &sample,
                                          std::ostream &log, std::string &error)
    {
        sample = ResonanceSample();
        error.clear();
        std::vector<std::string> hnames;
        if (!SystemPreparation::ResolveHamiltonian(plan, system, hnames, error))
            return false;
        SpinAPI::interaction_ptr field;
        if (!SystemPreparation::ResolveFieldInteraction(plan, system, field))
        {
            error = "no resonance Zeeman field interaction";
            return false;
        }
        auto zeeman = SystemPreparation::CollectZeemanInteractions(system, hnames);
        if (zeeman.empty())
            zeeman.push_back(field);
        FieldSyncGuard sync;
        if (plan.enforceZeemanSync)
            sync.Apply(zeeman, field->Field());
        const double fieldMagnitude = arma::norm(field->Field());
        if (!std::isfinite(fieldMagnitude) || fieldMagnitude <= 0)
        {
            error = "resonance field magnitude must be finite and positive";
            return false;
        }
        sample.field_mT = 1000 * fieldMagnitude;
        std::vector<SpinAPI::spin_ptr> spins;
        if (!SystemPreparation::ResolveDetectionSpins(plan, system, field, spins, sample.spin_names))
        {
            error = "could not resolve resonance detection spins";
            return false;
        }
        std::vector<ResonanceMagneticMomentTerm> terms;
        if (!SystemPreparation::PrepareMoments(spins, zeeman, terms, error))
            return false;
        SpinAPI::SpinSpace space(system);
        space.UseSuperoperatorSpace(false);
        space.UseFullTensorRotation(plan.fullTensorRotation);
        HS::HSPreparedState state;
        if (!SystemPreparation::PrepareState(plan, system, space, state, error))
            return false;
        const auto mz = Diagonalization::BuildMzBlocks(system->Spins());
        const bool useBlocks = plan.useMzBlocks && mz.blocks.size() > 1;
        std::vector<std::string> znames;
        for (const auto &z : zeeman)
            znames.push_back(z->Name());
        ResonanceOrientationGrid orientations;
        if (!OrientationSampling::Build(plan, system, field, hnames, orientations, log, error))
            return false;
        SpectrumRequest request;
        request.microwaveFrequencyGHz = plan.mwFrequencyGHz;
        request.linewidth_mT = plan.linewidth_mT;
        request.lineshape = plan.lineshape == "lorentzian" ? Lineshape::Lorentzian : Lineshape::Gaussian;
        std::vector<SpectrumPoint> points(orientations.grid.size());
        std::vector<std::string> errors(points.size());
        std::vector<int> success(points.size(), 1);

#pragma omp parallel
        {
            SpinAPI::SpinSpace local(system);
            local.UseSuperoperatorSpace(false);
            local.UseFullTensorRotation(plan.fullTensorRotation);
#pragma omp for schedule(static)
            for (size_t index = 0; index < orientations.grid.size(); ++index)
            {
                const auto &o = orientations.grid[index];
                auto &why = errors[index];
                for (int k = 0; k < orientations.gammaPoints; ++k)
                {
                    const double gamma = orientations.gammaPoints > 1
                                             ? 2 * arma::datum::pi * (k + .5) / orientations.gammaPoints
                                             : 0.;
                    arma::mat rotation;
                    OrientationSampling::Rotation(o.phi, o.theta, gamma, rotation);
                    arma::cx_mat density;
                    arma::sp_cx_mat h, hz;
                    if (!SystemPreparation::OrientState(local, state, rotation, density, why) ||
                        !SystemPreparation::BuildHamiltonian(local, hnames, rotation, h, why) ||
                        !SystemPreparation::BuildHamiltonian(local, znames, rotation, hz, why))
                    {
                        success[index] = 0;
                        break;
                    }
                    arma::vec energies;
                    arma::cx_mat vectors;
                    const bool diagonalized =
                        useBlocks && Diagonalization::IsBlockDiagonalMz(h, mz.mz2, 1e-12)
                            ? Diagonalization::EigSymBlockMz(h, mz.blocks, energies, vectors)
                            : arma::eig_sym(energies, vectors, arma::cx_mat(h));
                    if (!diagonalized)
                    {
                        why = "failed to diagonalize resonance Hamiltonian";
                        success[index] = 0;
                        break;
                    }
                    std::vector<ResonanceDetectionOperator> channels;
                    if (!ResonanceMagneticMomentBuilder::BuildTransverseChannels(
                            local, terms, rotation, plan.fullTensorRotation, channels, why))
                    {
                        success[index] = 0;
                        break;
                    }
                    arma::cx_mat mx(h.n_rows, h.n_cols, arma::fill::zeros), my = mx;
                    for (const auto &channel : channels)
                    {
                        mx += channel.x;
                        my += channel.y;
                    }
                    ResonanceLineSet lines;
                    SpectrumPoint point;
                    if (!ExactResonanceSolver::Generate(energies, vectors, density, hz / fieldMagnitude, mx,
                                                        my, request, lines, why, channels) ||
                        !ResonanceSpectrumEvaluator::Evaluate(lines, request, point, why))
                    {
                        success[index] = 0;
                        break;
                    }
                    ResonanceSpectrumProcessing::Accumulate(points[index], point,
                                                            o.weight * orientations.gammaWeight);
                }
            }
        }
        for (size_t i = 0; i < points.size(); ++i)
        {
            if (!success[i])
            {
                error = "orientation " + std::to_string(i) + ": " + errors[i];
                return false;
            }
            ResonanceSpectrumProcessing::Accumulate(sample.point, points[i], 1.);
        }
        return true;
    }
} // namespace RunSection::General::Resonance
