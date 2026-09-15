/////////////////////////////////////////////////////////////////////////
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "ResonanceCalculations.h"
#include "ResonanceOrientationSampler.h"
#include "ResonanceSystemPreparation.h"
#include "ResonanceDiagonalization.h"
#include "ResonanceFieldRoots.h"
#include "ExactResonanceSolver.h"
#include "ResonanceSpectrumEvaluator.h"
#include "SpinSystem.h"
#include "Spin.h"
#include "Interaction.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>

namespace RunSection::General::Resonance
{
    bool ResonanceExactCalculation::Sweep(const ResonanceExecutionPlan &plan,
                                          const SpinAPI::system_ptr &_system,
                                          const SpinAPI::interaction_ptr &_fieldInteraction,
                                          const ResonanceFieldSweep &sweep, ResonanceSpectrum &_cache,
                                          std::ostream &log, std::string &error)
    {
        error.clear();
        const auto &_field0 = sweep.field0;
        const auto &_fieldStep = sweep.fieldStep;

        // Cached sweep workflow:
        // - build the field axis once from the AddVector sweep,
        // - separate field-dependent Zeeman terms from the static Hamiltonian,
        // - evaluate transition moments for each powder orientation,
        // - deposit the resonances either directly or through the projection mesh.
        if (_system == nullptr || _fieldInteraction == nullptr)
            return false;

        const unsigned int steps = sweep.steps;
        if (steps < 2)
            return false;

        if (_field0.n_elem != 3 || !_field0.is_finite())
            return false;

        std::vector<double> field_T(steps, 0.0);
        _cache.field_mT.assign(steps, 0.0);
        _cache.total_x.assign(steps, 0.0);
        _cache.total_y.assign(steps, 0.0);
        _cache.total_perp.assign(steps, 0.0);
        _cache.cross_x.assign(steps, 0.0);
        _cache.cross_y.assign(steps, 0.0);
        _cache.steps = steps;

        arma::vec Bvec = _field0;
        for (unsigned int i = 0; i < steps; ++i)
        {
            const double Bmag = arma::norm(Bvec);
            if (!std::isfinite(Bmag) || Bmag <= 0.0)
                return false;
            field_T[i] = Bmag;
            _cache.field_mT[i] = 1.0e3 * Bmag;
            Bvec += _fieldStep;
        }

        const bool useResfieldsCache = (plan.sweepMode == ResonanceSweepMode::Projection);
        const bool useApproxCache = (!(plan.sweepMode == ResonanceSweepMode::Exact) && !useResfieldsCache);
        // Cache modes:
        // - exact: diagonalize at every output field point.
        // - approx: locate resonances on a coarser scan and broaden them directly.
        // - resonanceprojection: locate resonance fields first and, when possible,
        //   project the orientation mesh continuously onto the output field axis.
        const double dBstep = (steps > 1) ? (field_T[1] - field_T[0]) : 0.0;
        const double dBabs = std::abs(dBstep);
        if ((plan.sweepMode == ResonanceSweepMode::RefinedRoots) && dBabs == 0.0)
            return false;
        if ((plan.sweepMode == ResonanceSweepMode::RefinedRoots))
        {
            // Norms of a vector sweep that crosses B=0 are not an affine
            // field axis. Refuse it rather than assigning incorrect roots.
            for (unsigned i = 1; i < steps; ++i)
                if (std::abs(field_T[i] - field_T[i - 1] - dBstep) > 1e-10 * std::max(1., dBabs))
                    return false;
        }

        const double omega_mw = 2.0 * arma::datum::pi * plan.mwFrequencyGHz;

        SpinAPI::SpinSpace space(*_system);
        space.UseSuperoperatorSpace(false);
        space.UseFullTensorRotation(plan.fullTensorRotation);
        const arma::uword spaceDim = space.HilbertSpaceDimensions();
        const auto allSpins = _system->Spins();
        const MzBlocks mzBlocks = Diagonalization::BuildMzBlocks(allSpins);
        const bool hasMzBlocks =
            (plan.useMzBlocks && !mzBlocks.mz2.empty() &&
             mzBlocks.mz2.size() == static_cast<size_t>(spaceDim) && mzBlocks.blocks.size() > 1);

        std::vector<std::string> h0list;
        if (!SystemPreparation::ResolveHamiltonian(plan, _system, h0list, error))
            return false;

        std::vector<SpinAPI::interaction_ptr> zeemanInteractions =
            SystemPreparation::CollectZeemanInteractions(_system, h0list);
        if (zeemanInteractions.empty() && _fieldInteraction != nullptr)
            zeemanInteractions.push_back(_fieldInteraction);

        auto isZeemanName = [&](const std::string &name) -> bool
        {
            for (const auto &inter : zeemanInteractions)
            {
                if (inter != nullptr && inter->Name() == name)
                    return true;
            }
            return false;
        };

        std::vector<std::string> h0list_noB;
        h0list_noB.reserve(h0list.size());
        for (const auto &name : h0list)
        {
            if (!isZeemanName(name))
                h0list_noB.push_back(name);
        }

        std::vector<SpinAPI::spin_ptr> detectSpins;
        std::vector<std::string> detectSpinNames;
        if (!SystemPreparation::ResolveDetectionSpins(plan, _system, _fieldInteraction, detectSpins,
                                                      detectSpinNames))
            return false;
        if (detectSpins.empty())
            return false;
        const OrientationDiagnostics orientationDebug = plan.diagnostics;
        if (orientationDebug.enabled)
            orientationDebug.InitialiseFile(_system->Name(), detectSpinNames);
        std::ofstream debugOut;
        if (orientationDebug.enabled)
            debugOut.open(orientationDebug.file, std::ios::out | std::ios::app);
        _cache.spin_names = detectSpinNames;
        _cache.spin_x.assign(detectSpins.size(), std::vector<double>(steps, 0.0));
        _cache.spin_y.assign(detectSpins.size(), std::vector<double>(steps, 0.0));
        _cache.spin_perp.assign(detectSpins.size(), std::vector<double>(steps, 0.0));
        _cache.spin_p.assign(detectSpins.size(), std::vector<double>(steps, 0.0));
        _cache.spin_m.assign(detectSpins.size(), std::vector<double>(steps, 0.0));

        HS::HSPreparedState preparedState;
        if (!SystemPreparation::PrepareState(plan, _system, space, preparedState, error))
            return false;
        const bool useOrientationThermal = preparedState.orientationSpecificThermal;
        const auto &thermalhamiltonian_list = preparedState.thermalHamiltonian;
        const double thermalTemperature = preparedState.thermalTemperature;

        ResonanceOrientationGrid orientations;
        if (!OrientationSampling::Build(plan, _system, _fieldInteraction, h0list, orientations, log, error))
            return false;
        const auto &grid = orientations.grid;
        const int numPoints = static_cast<int>(grid.size());
        const bool useSopheGrid = orientations.usesSophe, haveSopheParams = orientations.hasSopheParameters;
        const auto &sopheParams = orientations.sophe;
        const int sopheGridSize = orientations.gridSize, gamma_points = orientations.gammaPoints;
        const double gamma_weight = orientations.gammaWeight;

        const double lwB_mT = std::abs(plan.linewidth_mT);
        General::Resonance::SpectrumRequest resonanceRequest;
        resonanceRequest.microwaveFrequencyGHz = plan.mwFrequencyGHz;
        resonanceRequest.linewidth_mT = lwB_mT;
        resonanceRequest.lineshape = (plan.lineshape == "lorentzian")
                                         ? General::Resonance::Lineshape::Lorentzian
                                         : General::Resonance::Lineshape::Gaussian;
        resonanceRequest.populationThreshold = 1.0e-15;
        resonanceRequest.minimumSlope = 1.0e-15;
        resonanceRequest.maximumDBdOmega = 1.0e5;

        double lineWindow = 0.0;
        if (useApproxCache || useResfieldsCache)
        {
            const double lwB_T = lwB_mT * 1.0e-3;
            lineWindow = (lwB_T > 0.0 && dBabs > 0.0) ? 6.0 * lwB_T : 0.0;
        }

        std::vector<ResonanceMagneticMomentTerm> momentTerms;
        if (!SystemPreparation::PrepareMoments(detectSpins, zeemanInteractions, momentTerms, error))
            return false;

        std::vector<std::string> zeelist;
        zeelist.reserve(zeemanInteractions.size());
        for (const auto &inter : zeemanInteractions)
            zeelist.push_back(inter->Name());

        const arma::uword dim = space.HilbertSpaceDimensions();
        std::vector<std::pair<arma::uword, arma::uword>> transitions;
        transitions.reserve(static_cast<size_t>(dim) * static_cast<size_t>(dim - 1) / 2);
        for (arma::uword m = 0; m < dim; ++m)
        {
            for (arma::uword n = m + 1; n < dim; ++n)
                transitions.emplace_back(m, n);
        }

        const arma::cx_double I(0.0, 1.0);
        const size_t spin_count = detectSpins.size();
        const bool projectionCandidate = useSopheGrid && haveSopheParams && numPoints > 1 && lwB_mT <= 0.0 &&
                                         (useApproxCache || useResfieldsCache);
        SpinAPI::PowderProjectionMesh projectionMesh;
        bool projectionPossible = false;
        size_t projectionSamples = 0;
        std::vector<unsigned int> projectionCounts;
        std::vector<double> projectionPositions;
        std::vector<double> projectionTotalX;
        std::vector<double> projectionTotalY;
        std::vector<double> projectionTotalPerp;
        std::vector<double> projectionCrossX;
        std::vector<double> projectionCrossY;
        std::vector<std::vector<double>> projectionSpinX;
        std::vector<std::vector<double>> projectionSpinY;
        std::vector<std::vector<double>> projectionSpinPerp;
        std::vector<std::vector<double>> projectionSpinP;
        std::vector<std::vector<double>> projectionSpinM;
        if (projectionCandidate)
        {
            projectionPossible = SpinAPI::BuildSopheProjectionMesh(
                sopheParams.nOctants, sopheParams.closedPhi, sopheGridSize, grid, projectionMesh);
            if (projectionPossible)
            {
                projectionSamples = static_cast<size_t>(numPoints) * static_cast<size_t>(gamma_points);
                const size_t projectionSize = transitions.size() * projectionSamples;
                const double nan = std::numeric_limits<double>::quiet_NaN();
                projectionCounts.assign(projectionSize, 0U);
                projectionPositions.assign(projectionSize, nan);
                projectionTotalX.assign(projectionSize, 0.0);
                projectionTotalY.assign(projectionSize, 0.0);
                projectionTotalPerp.assign(projectionSize, 0.0);
                projectionCrossX.assign(projectionSize, 0.0);
                projectionCrossY.assign(projectionSize, 0.0);
                projectionSpinX.assign(spin_count, std::vector<double>(projectionSize, 0.0));
                projectionSpinY.assign(spin_count, std::vector<double>(projectionSize, 0.0));
                projectionSpinPerp.assign(spin_count, std::vector<double>(projectionSize, 0.0));
                projectionSpinP.assign(spin_count, std::vector<double>(projectionSize, 0.0));
                projectionSpinM.assign(spin_count, std::vector<double>(projectionSize, 0.0));
            }
        }

        const std::vector<double> *field_scan = &field_T;
        std::vector<double> field_scan_storage;
        unsigned int scan_steps = steps;
        if (useResfieldsCache)
        {
            // The resonance scan only needs to resolve sign changes in the transition
            // detuning, so it can use a coarser field mesh than the final output axis.
            unsigned int points = (plan.sweepCacheResfieldPoints > 1)
                                      ? static_cast<unsigned int>(plan.sweepCacheResfieldPoints)
                                      : 0U;
            if (points < 2)
            {
                points = std::min<unsigned int>(steps, 80U);
            }
            if (points < 2)
                points = 2;

            field_scan_storage.resize(points);
            const double Bmin = field_T.front();
            const double Bmax = field_T.back();
            const double step = (points > 1) ? ((Bmax - Bmin) / static_cast<double>(points - 1)) : 0.0;
            for (unsigned int i = 0; i < points; ++i)
            {
                field_scan_storage[i] = Bmin + step * static_cast<double>(i);
            }
            field_scan = &field_scan_storage;
            scan_steps = points;
        }

        for (int grid_num = 0; grid_num < numPoints; ++grid_num)
        {
            auto [theta, phi, w_solid] = grid[grid_num];
            const double base_weight = w_solid;

            for (int gamma_idx = 0; gamma_idx < gamma_points; ++gamma_idx)
            {
                double gamma = 0.0;
                if (gamma_points > 1)
                    gamma = 2.0 * arma::datum::pi * (static_cast<double>(gamma_idx) + 0.5) /
                            static_cast<double>(gamma_points);

                const double w = base_weight * gamma_weight;

                arma::mat Rot;
                if (!OrientationSampling::Rotation(phi, theta, gamma, Rot))
                {
                    error = "invalid powder rotation";
                    return false;
                }

                arma::cx_mat rho_oriented;
                arma::cx_mat Hthermal_static, Hthermal_dHdB;
                if (useOrientationThermal)
                {
                    // Equilibrium populations follow the sweep field. Preserve the
                    // explicitly selected thermal Hamiltonian, which can differ
                    // from the Hamiltonian used to calculate resonance energies.
                    std::vector<std::string> thermalStaticNames, thermalZeemanNames;
                    for (const auto &name : thermalhamiltonian_list)
                        (isZeemanName(name) ? thermalZeemanNames : thermalStaticNames).push_back(name);
                    arma::sp_cx_mat thermalStatic, thermalZeeman;
                    if (!SystemPreparation::BuildHamiltonian(space, thermalStaticNames, Rot, thermalStatic,
                                                             error) ||
                        !SystemPreparation::BuildHamiltonian(space, thermalZeemanNames, Rot, thermalZeeman,
                                                             error))
                        return false;
                    Hthermal_static = arma::cx_mat(thermalStatic);
                    Hthermal_dHdB = arma::cx_mat(thermalZeeman) / field_T.front();
                }
                else if (!SystemPreparation::OrientState(space, preparedState, Rot, rho_oriented, error))
                    return false;

                // Cached Resonance spectra still follows the pepper-style Hilbert
                // formulation. Splitting Hstatic and Hz only accelerates
                // the field scan; it must remain the full rotated ZYZ
                // Hamiltonian, not the secular H0/H1 propagation helper.
                arma::sp_cx_mat Hstatic_sp;
                if (h0list_noB.empty())
                {
                    Hstatic_sp = arma::zeros<arma::sp_cx_mat>(dim, dim);
                }
                else if (!SystemPreparation::BuildHamiltonian(space, h0list_noB, Rot, Hstatic_sp, error))
                {
                    return false;
                }

                arma::sp_cx_mat Hz_sp;
                if (!SystemPreparation::BuildHamiltonian(space, zeelist, Rot, Hz_sp, error))
                    return false;
                const bool can_block = hasMzBlocks &&
                                       Diagonalization::IsBlockDiagonalMz(Hstatic_sp, mzBlocks.mz2, 1e-12) &&
                                       Diagonalization::IsBlockDiagonalMz(Hz_sp, mzBlocks.mz2, 1e-12);
                arma::cx_mat Hstatic = arma::cx_mat(Hstatic_sp);
                arma::cx_mat Hz = arma::cx_mat(Hz_sp);
                const double invField0 = 1.0 / field_T[0];
                arma::cx_mat dHdB = Hz * invField0;
                const arma::sp_cx_mat dHdB_sp = Hz_sp * invField0;

                std::vector<ResonanceDetectionOperator> detectionChannels;
                if (!ResonanceMagneticMomentBuilder::BuildTransverseChannels(
                        space, momentTerms, Rot, plan.fullTensorRotation, detectionChannels, error))
                    return false;
                std::vector<arma::cx_mat> mux_list(spin_count), muy_list(spin_count);
                arma::cx_mat muxT(spaceDim, spaceDim, arma::fill::zeros), muyT = muxT;
                for (size_t i = 0; i < spin_count; ++i)
                {
                    mux_list[i] = detectionChannels[i].x;
                    muy_list[i] = detectionChannels[i].y;
                    muxT += mux_list[i];
                    muyT += muy_list[i];
                }

                if ((plan.sweepMode == ResonanceSweepMode::RefinedRoots))
                {
                    // Full electronic diagonalization at true resonance fields.
                    // Keep magnetic moments and line weighting in General Resonance;
                    // the task only accumulates field profiles and powder weights.

                    const auto &channels = detectionChannels;
                    const auto &mx = muxT;
                    const auto &my = muyT;
                    std::string rootError;
                    std::vector<ResonanceFieldRoot> roots;
                    const double padding = 6e-3 * lwB_mT;
                    const double low = std::max(1e-12, std::min(field_T.front(), field_T.back()) - padding);
                    const double high = std::max(field_T.front(), field_T.back()) + padding;
                    if (!ResonanceFieldRoots::Locate(Hstatic, dHdB, low, high, omega_mw, roots, rootError))
                    {
                        log << "Refined resonance fields failed: " << rootError << std::endl;
                        return false;
                    }
                    const double centerProfile =
                        ResonanceSpectrumProcessing::LineshapeValue(plan, 0., lwB_mT);
                    for (const auto &root : roots)
                    {
                        if (useOrientationThermal &&
                            !space.ThermalStateFromHamiltonian(Hthermal_static + root.fieldT * Hthermal_dHdB,
                                                               thermalTemperature, rho_oriented))
                            return false;
                        ResonanceLineSet lines;
                        if (!ExactResonanceSolver::Generate(arma::sp_cx_mat(Hstatic + root.fieldT * dHdB),
                                                            rho_oriented, dHdB_sp, mx, my, resonanceRequest,
                                                            lines, rootError, channels))
                            return false;
                        for (const auto &line : lines.lines)
                        {
                            if (line.lower != root.lower || line.upper != root.upper)
                                continue;
                            ResonanceLineSet single;
                            single.fieldJacobianQualified = true;
                            single.lines.push_back(line);
                            SpectrumPoint peak;
                            if (!ResonanceSpectrumEvaluator::Evaluate(single, resonanceRequest, peak,
                                                                      rootError))
                                return false;
                            const int center =
                                static_cast<int>(std::llround((root.fieldT - field_T.front()) / dBstep));
                            const int half = static_cast<int>(std::ceil(padding / dBabs)) + 2;
                            const int first = std::max(0, center - half),
                                      last = std::min(static_cast<int>(steps) - 1, center + half);
                            for (int j = first; j <= last; ++j)
                            {
                                const double weight = w *
                                                      ResonanceSpectrumProcessing::LineshapeValue(
                                                          plan, 1e3 * (field_T[j] - root.fieldT), lwB_mT) /
                                                      centerProfile;
                                _cache.total_x[j] += weight * peak.totalX;
                                _cache.total_y[j] += weight * peak.totalY;
                                _cache.total_perp[j] += weight * peak.totalPerpendicular;
                                _cache.cross_x[j] += weight * peak.crossX;
                                _cache.cross_y[j] += weight * peak.crossY;
                                for (size_t k = 0; k < spin_count; ++k)
                                {
                                    _cache.spin_x[k][j] += weight * peak.channels[k].x;
                                    _cache.spin_y[k][j] += weight * peak.channels[k].y;
                                    _cache.spin_perp[k][j] += weight * peak.channels[k].perpendicular;
                                    _cache.spin_p[k][j] += weight * peak.channels[k].plus;
                                    _cache.spin_m[k][j] += weight * peak.channels[k].minus;
                                }
                            }
                        }
                    }
                    continue;
                }

                if (useApproxCache || useResfieldsCache)
                {
                    // Track each transition detuning along the scan and detect zero
                    // crossings. Each zero crossing corresponds to a resonance field.
                    std::vector<double> prev_delta(transitions.size(), 0.0);
                    arma::cx_mat prev_eigvec;
                    bool have_prev = false;

                    for (unsigned int step = 0; step < scan_steps; ++step)
                    {
                        arma::vec eigval;
                        arma::cx_mat eigvec;
                        bool have_eig = false;
                        const double Bscan = (*field_scan)[step];
                        if (can_block)
                        {
                            const double scale = Bscan * invField0;
                            arma::sp_cx_mat H_sp = Hstatic_sp + scale * Hz_sp;
                            have_eig = Diagonalization::EigSymBlockMz(H_sp, mzBlocks.blocks, eigval, eigvec);
                        }
                        else
                        {
                            arma::cx_mat H = Hstatic + Bscan * dHdB;
                            have_eig = arma::eig_sym(eigval, eigvec, H);
                        }
                        if (!have_eig)
                        {
                            error = "failed to diagonalize a sweep Hamiltonian";
                            return false;
                        }

                        arma::cx_mat dHdB_ev = dHdB * eigvec;
                        arma::vec dHdB_diag(dim);
                        for (arma::uword i = 0; i < dim; ++i)
                        {
                            dHdB_diag(i) = std::real(arma::cdot(eigvec.col(i), dHdB_ev.col(i)));
                        }

                        std::vector<double> curr_delta(transitions.size(), 0.0);
                        for (size_t t = 0; t < transitions.size(); ++t)
                        {
                            const auto [m, n] = transitions[t];
                            curr_delta[t] = (eigval(n) - eigval(m)) - omega_mw;
                        }

                        if (!have_prev)
                        {
                            prev_delta = curr_delta;
                            prev_eigvec = eigvec;
                            have_prev = true;
                            continue;
                        }

                        for (size_t t = 0; t < transitions.size(); ++t)
                        {
                            const double d1 = prev_delta[t];
                            const double d2 = curr_delta[t];
                            if (d1 == 0.0 && d2 == 0.0)
                                continue;
                            if (d1 == 0.0 || d2 == 0.0 || (d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
                            {
                                const double denom = (d1 - d2);
                                if (std::abs(denom) < 1e-15)
                                    continue;
                                const double tfrac = d1 / denom;
                                const double Bres = (*field_scan)[step - 1] +
                                                    tfrac * ((*field_scan)[step] - (*field_scan)[step - 1]);

                                const bool use_prev = std::abs(d1) <= std::abs(d2);
                                const arma::cx_mat &eigvec_use = use_prev ? prev_eigvec : eigvec;

                                const auto [m, n] = transitions[t];
                                const arma::cx_vec Um = eigvec_use.col(m);
                                const arma::cx_vec Vn = eigvec_use.col(n);

                                if (useOrientationThermal &&
                                    !space.ThermalStateFromHamiltonian(Hthermal_static + Bres * Hthermal_dHdB,
                                                                       thermalTemperature, rho_oriented))
                                {
                                    error = "failed to prepare resonance-field thermal populations";
                                    return false;
                                }
                                const double population = std::real(arma::cdot(Um, rho_oriented * Um)) -
                                                          std::real(arma::cdot(Vn, rho_oriented * Vn));

                                if (std::abs(population) < 1e-15)
                                    continue;

                                const double abs_domega_dB = std::abs(dHdB_diag(n) - dHdB_diag(m));
                                if (!std::isfinite(abs_domega_dB) || abs_domega_dB < 1e-15)
                                    continue;
                                const double dBdE = 1.0 / abs_domega_dB;
                                if (dBdE > 1e5)
                                    continue;

                                std::vector<double> amp_spin_x(spin_count, 0.0);
                                std::vector<double> amp_spin_y(spin_count, 0.0);
                                std::vector<double> amp_spin_perp(spin_count, 0.0);
                                std::vector<double> amp_spin_p(spin_count, 0.0);
                                std::vector<double> amp_spin_m(spin_count, 0.0);

                                arma::cx_double muTx(0.0, 0.0);
                                arma::cx_double muTy(0.0, 0.0);
                                double I_sum_x = 0.0;
                                double I_sum_y = 0.0;

                                for (size_t i = 0; i < spin_count; ++i)
                                {
                                    const arma::cx_double muix = arma::cdot(Um, mux_list[i] * Vn);
                                    const arma::cx_double muiy = arma::cdot(Um, muy_list[i] * Vn);
                                    muTx += muix;
                                    muTy += muiy;

                                    const double Iix = std::norm(muix);
                                    const double Iiy = std::norm(muiy);
                                    I_sum_x += Iix;
                                    I_sum_y += Iiy;

                                    amp_spin_x[i] = population * Iix * dBdE;
                                    amp_spin_y[i] = population * Iiy * dBdE;
                                    amp_spin_perp[i] = population * 0.5 * (Iix + Iiy) * dBdE;

                                    const arma::cx_double mup = muix + I * muiy;
                                    const arma::cx_double mum = muix - I * muiy;
                                    amp_spin_p[i] = population * std::norm(mup) * dBdE;
                                    amp_spin_m[i] = population * std::norm(mum) * dBdE;
                                }

                                const double ITx = std::norm(muTx);
                                const double ITy = std::norm(muTy);
                                const double ICx = ITx - I_sum_x;
                                const double ICy = ITy - I_sum_y;

                                const double amp_total_x = population * ITx * dBdE;
                                const double amp_total_y = population * ITy * dBdE;
                                const double amp_total_perp = population * 0.5 * (ITx + ITy) * dBdE;
                                const double amp_crossx = population * ICx * dBdE;
                                const double amp_crossy = population * ICy * dBdE;

                                if (orientationDebug.ShouldRecord(grid_num, Bres) && debugOut.good())
                                {
                                    debugOut << _system->Name() << "\t" << grid_num << "\t" << gamma_idx
                                             << "\t" << std::setprecision(12) << theta << "\t" << phi << "\t"
                                             << gamma << "\t" << w << "\t" << m << "\t" << n << "\t" << Bres
                                             << "\t" << (1.0e3 * Bres) << "\t" << amp_total_x << "\t"
                                             << amp_total_y << "\t" << amp_total_perp << "\t" << amp_crossx
                                             << "\t" << amp_crossy;
                                    for (size_t i = 0; i < spin_count; ++i)
                                    {
                                        debugOut << "\t" << amp_spin_x[i] << "\t" << amp_spin_y[i] << "\t"
                                                 << amp_spin_perp[i] << "\t" << amp_spin_p[i] << "\t"
                                                 << amp_spin_m[i];
                                    }
                                    debugOut << "\n";
                                }

                                if (projectionPossible)
                                {
                                    // Store one resonance field per transition and
                                    // orientation sample. These values are projected
                                    // onto the final sweep axis after the powder loop.
                                    const size_t sampleIndex =
                                        static_cast<size_t>(gamma_idx) * static_cast<size_t>(numPoints) +
                                        static_cast<size_t>(grid_num);
                                    const size_t projectionIndex = t * projectionSamples + sampleIndex;
                                    if (projectionCounts[projectionIndex] == 0U)
                                    {
                                        projectionPositions[projectionIndex] = 1.0e3 * Bres;
                                        projectionTotalX[projectionIndex] = amp_total_x;
                                        projectionTotalY[projectionIndex] = amp_total_y;
                                        projectionTotalPerp[projectionIndex] = amp_total_perp;
                                        projectionCrossX[projectionIndex] = amp_crossx;
                                        projectionCrossY[projectionIndex] = amp_crossy;
                                        for (size_t i = 0; i < spin_count; ++i)
                                        {
                                            projectionSpinX[i][projectionIndex] = amp_spin_x[i];
                                            projectionSpinY[i][projectionIndex] = amp_spin_y[i];
                                            projectionSpinPerp[i][projectionIndex] = amp_spin_perp[i];
                                            projectionSpinP[i][projectionIndex] = amp_spin_p[i];
                                            projectionSpinM[i][projectionIndex] = amp_spin_m[i];
                                        }
                                    }
                                    else
                                    {
                                        projectionPossible = false;
                                    }
                                    projectionCounts[projectionIndex] += 1U;
                                }

                                if (lwB_mT <= 0.0 || dBabs == 0.0)
                                {
                                    size_t idx = 0;
                                    if (dBstep != 0.0)
                                    {
                                        const double pos = (Bres - field_T[0]) / dBstep;
                                        idx = static_cast<size_t>(std::llround(pos));
                                    }

                                    if (idx < steps)
                                    {
                                        _cache.total_x[idx] += w * amp_total_x;
                                        _cache.total_y[idx] += w * amp_total_y;
                                        _cache.total_perp[idx] += w * amp_total_perp;
                                        _cache.cross_x[idx] += w * amp_crossx;
                                        _cache.cross_y[idx] += w * amp_crossy;

                                        for (size_t i = 0; i < spin_count; ++i)
                                        {
                                            _cache.spin_x[i][idx] += w * amp_spin_x[i];
                                            _cache.spin_y[i][idx] += w * amp_spin_y[i];
                                            _cache.spin_perp[i][idx] += w * amp_spin_perp[i];
                                            _cache.spin_p[i][idx] += w * amp_spin_p[i];
                                            _cache.spin_m[i][idx] += w * amp_spin_m[i];
                                        }
                                    }
                                }
                                else
                                {
                                    int start = 0;
                                    int end = static_cast<int>(steps) - 1;
                                    if (lineWindow > 0.0 && dBabs > 0.0)
                                    {
                                        const int half = static_cast<int>(std::ceil(lineWindow / dBabs));
                                        const double pos = (Bres - field_T[0]) / dBstep;
                                        const int center = static_cast<int>(std::llround(pos));
                                        start = center - half;
                                        end = center + half;

                                        if (end < 0 || start >= static_cast<int>(steps))
                                            continue;

                                        start = std::max(0, start);
                                        end = std::min(static_cast<int>(steps) - 1, end);
                                    }

                                    for (int j = start; j <= end; ++j)
                                    {
                                        const double deltaB_mT =
                                            (field_T[static_cast<size_t>(j)] - Bres) * 1.0e3;
                                        const double L = ResonanceSpectrumProcessing::LineshapeValue(
                                            plan, deltaB_mT, lwB_mT);
                                        if (L == 0.0)
                                            continue;

                                        const double weight = w * L;
                                        _cache.total_x[static_cast<size_t>(j)] += weight * amp_total_x;
                                        _cache.total_y[static_cast<size_t>(j)] += weight * amp_total_y;
                                        _cache.total_perp[static_cast<size_t>(j)] += weight * amp_total_perp;
                                        _cache.cross_x[static_cast<size_t>(j)] += weight * amp_crossx;
                                        _cache.cross_y[static_cast<size_t>(j)] += weight * amp_crossy;

                                        for (size_t i = 0; i < spin_count; ++i)
                                        {
                                            _cache.spin_x[i][static_cast<size_t>(j)] +=
                                                weight * amp_spin_x[i];
                                            _cache.spin_y[i][static_cast<size_t>(j)] +=
                                                weight * amp_spin_y[i];
                                            _cache.spin_perp[i][static_cast<size_t>(j)] +=
                                                weight * amp_spin_perp[i];
                                            _cache.spin_p[i][static_cast<size_t>(j)] +=
                                                weight * amp_spin_p[i];
                                            _cache.spin_m[i][static_cast<size_t>(j)] +=
                                                weight * amp_spin_m[i];
                                        }
                                    }
                                }
                            }
                        }

                        prev_delta.swap(curr_delta);
                        prev_eigvec = eigvec;
                    }
                }
                else
                {

                    for (unsigned int step = 0; step < steps; ++step)
                    {
                        arma::vec eigval;
                        arma::cx_mat eigvec;
                        bool have_eig = false;
                        if (can_block)
                        {
                            const double scale = field_T[step] * invField0;
                            arma::sp_cx_mat H_sp = Hstatic_sp + scale * Hz_sp;
                            have_eig = Diagonalization::EigSymBlockMz(H_sp, mzBlocks.blocks, eigval, eigvec);
                        }
                        else
                        {
                            arma::cx_mat H = Hstatic + field_T[step] * dHdB;
                            have_eig = arma::eig_sym(eigval, eigvec, H);
                        }
                        if (!have_eig)
                        {
                            error = "failed to diagonalize a sweep Hamiltonian";
                            return false;
                        }

                        if (useOrientationThermal && !space.ThermalStateFromHamiltonian(
                                                         Hthermal_static + field_T[step] * Hthermal_dHdB,
                                                         thermalTemperature, rho_oriented))
                        {
                            error = "failed to prepare sweep thermal populations";
                            return false;
                        }

                        General::Resonance::ResonanceLineSet resonanceLines;
                        std::string resonanceError;
                        if (!General::Resonance::ExactResonanceSolver::Generate(
                                eigval, eigvec, rho_oriented, dHdB_sp, muxT, muyT, resonanceRequest,
                                resonanceLines, resonanceError, detectionChannels))
                        {
                            error = resonanceError;
                            return false;
                        }

                        General::Resonance::SpectrumPoint resonancePoint;
                        if (!General::Resonance::ResonanceSpectrumEvaluator::Evaluate(
                                resonanceLines, resonanceRequest, resonancePoint, resonanceError))
                        {
                            error = resonanceError;
                            return false;
                        }
                        if (resonancePoint.channels.size() != spin_count)
                        {
                            error = "inconsistent resonance detection channel count";
                            return false;
                        }

                        _cache.total_x[step] += w * resonancePoint.totalX;
                        _cache.total_y[step] += w * resonancePoint.totalY;
                        _cache.total_perp[step] += w * resonancePoint.totalPerpendicular;
                        _cache.cross_x[step] += w * resonancePoint.crossX;
                        _cache.cross_y[step] += w * resonancePoint.crossY;

                        for (size_t i = 0; i < spin_count; ++i)
                        {
                            const auto &channel = resonancePoint.channels[i];
                            _cache.spin_x[i][step] += w * channel.x;
                            _cache.spin_y[i][step] += w * channel.y;
                            _cache.spin_perp[i][step] += w * channel.perpendicular;
                            _cache.spin_p[i][step] += w * channel.plus;
                            _cache.spin_m[i][step] += w * channel.minus;
                        }
                    }
                }
            }
        }

        if (projectionPossible)
        {
            // Replace the provisional binned cache by the continuously projected
            // spectrum assembled from resonance fields and the powder-orientation mesh.
            auto projectChannel = [&](const std::vector<double> &amplitudeData, std::vector<double> &target)
            {
                std::fill(target.begin(), target.end(), 0.0);
                std::vector<double> positions(static_cast<size_t>(numPoints),
                                              std::numeric_limits<double>::quiet_NaN());
                std::vector<double> amplitudes(static_cast<size_t>(numPoints), 0.0);
                std::vector<double> local(target.size(), 0.0);

                for (int gamma_idx = 0; gamma_idx < gamma_points; ++gamma_idx)
                {
                    for (size_t t = 0; t < transitions.size(); ++t)
                    {
                        bool hasResonance = false;
                        for (int grid_num = 0; grid_num < numPoints; ++grid_num)
                        {
                            const size_t sampleIndex =
                                static_cast<size_t>(gamma_idx) * static_cast<size_t>(numPoints) +
                                static_cast<size_t>(grid_num);
                            const size_t projectionIndex = t * projectionSamples + sampleIndex;
                            positions[static_cast<size_t>(grid_num)] = projectionPositions[projectionIndex];
                            amplitudes[static_cast<size_t>(grid_num)] = amplitudeData[projectionIndex];
                            hasResonance =
                                hasResonance || std::isfinite(positions[static_cast<size_t>(grid_num)]);
                        }

                        if (!hasResonance)
                            continue;

                        std::fill(local.begin(), local.end(), 0.0);
                        if (projectionMesh.axial)
                            SpinAPI::ProjectPowderZones(positions, amplitudes, projectionMesh.weights,
                                                        _cache.field_mT, local);
                        else
                            SpinAPI::ProjectPowderTriangles(projectionMesh.triangles, projectionMesh.weights,
                                                            positions, amplitudes, _cache.field_mT, local);

                        for (size_t j = 0; j < target.size(); ++j)
                            target[j] += gamma_weight * local[j];
                    }
                }
            };

            projectChannel(projectionTotalX, _cache.total_x);
            projectChannel(projectionTotalY, _cache.total_y);
            projectChannel(projectionTotalPerp, _cache.total_perp);
            projectChannel(projectionCrossX, _cache.cross_x);
            projectChannel(projectionCrossY, _cache.cross_y);
            for (size_t i = 0; i < spin_count; ++i)
            {
                projectChannel(projectionSpinX[i], _cache.spin_x[i]);
                projectChannel(projectionSpinY[i], _cache.spin_y[i]);
                projectChannel(projectionSpinPerp[i], _cache.spin_perp[i]);
                projectChannel(projectionSpinP[i], _cache.spin_p[i]);
                projectChannel(projectionSpinM[i], _cache.spin_m[i]);
            }
        }

        ResonanceSpectrumProcessing::ApplyDetectionHarmonic(plan, _cache);

        return true;
    }
} // namespace RunSection::General::Resonance
