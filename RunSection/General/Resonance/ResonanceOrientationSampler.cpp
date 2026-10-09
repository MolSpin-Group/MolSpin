/////////////////////////////////////////////////////////////////////////
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "ResonanceOrientationSampler.h"
#include "ResonanceSystemPreparation.h"
#include "SpinSystem.h"
#include "Spin.h"
#include "Interaction.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>

namespace RunSection::General::Resonance
{
    namespace OrientationSampling
    {
        bool Rotation(double alpha, double beta, double gamma, arma::mat &rotation)
        {
            // Explicit adapter from the established resonance passive ZYZ convention
            // to SpinAPI's active ZYZ primitive. No duplicate rotation formula.
            return SpinAPI::CreateZYZRotationMatrix(-gamma, -beta, -alpha, rotation);
        }
        arma::mat PassiveZYZRotation(const arma::vec &angles)
        {
            arma::mat result;
            Rotation(angles.n_elem > 0 ? angles(0) : 0., angles.n_elem > 1 ? angles(1) : 0.,
                     angles.n_elem > 2 ? angles(2) : 0., result);
            return result;
        }
        bool Build(const ResonanceExecutionPlan &plan, const SpinAPI::system_ptr &system,
                   const SpinAPI::interaction_ptr &field, const std::vector<std::string> &hamiltonian,
                   ResonanceOrientationGrid &out, std::ostream &log, std::string &error)
        {
            out = ResonanceOrientationGrid();
            error.clear();
            out.usesSophe = plan.powderGridType == "sophe";
            if (out.usesSophe)
            {
                std::string symmetry = plan.powderGridSymmetry;
                const auto lower = ResonanceLowercase(symmetry);
                if (lower.empty() || lower == "auto" || lower == "automatic")
                    symmetry = AutoDetectSopheSymmetry(system, field, hamiltonian, plan.fullTensorRotation);
                out.hasSopheParameters = SpinAPI::GetSopheGridParameters(symmetry, out.sophe);
                out.gridSize = plan.powderGridSize;
                if (out.gridSize < 2 && plan.powdersamplingpoints > 1 && out.hasSopheParameters)
                {
                    int best = std::numeric_limits<int>::max();
                    for (int size = 2; size <= 200; ++size)
                    {
                        const int count =
                            SpinAPI::SopheGridPointCount(size, out.sophe.nOctants, out.sophe.closedPhi);
                        const int difference = std::abs(count - plan.powdersamplingpoints);
                        if (difference < best)
                        {
                            best = difference;
                            out.gridSize = size;
                        }
                        if (difference == 0)
                            break;
                    }
                }
                if (out.gridSize < 2)
                    out.gridSize = 19;
                if (!SpinAPI::CreateSophePowderGrid(out.gridSize, symmetry, out.grid))
                {
                    error = "failed to construct SOPHE resonance grid";
                    return false;
                }
                log << "Resonance SOPHE " << symmetry << ", GridSize=" << out.gridSize
                    << ", orientations=" << out.grid.size() << std::endl;
            }
            else if (plan.powdersamplingpoints > 1)
            {
                if (!SpinAPI::CreateUniformPowderGrid(plan.powdersamplingpoints,
                                                      plan.powderFullSphere
                                                          ? SpinAPI::PowderGridDomain::FullSphere
                                                          : SpinAPI::PowderGridDomain::UpperHemisphere,
                                                      out.grid))
                {
                    error = "failed to construct uniform resonance grid";
                    return false;
                }
            }
            else
                out.grid.push_back({0., 0., 1.});
            out.gammaPoints = out.grid.size() > 1 ? std::max(1, plan.powderGammaPoints) : 1;
            // Preserve the resonance task's raw integral measure for numerical parity.
            // General propagation uses unit ensemble averages; this adapter is explicit.
            out.gammaWeight = (out.usesSophe ? 2 * arma::datum::pi : 1.) / out.gammaPoints;
            return true;
        }
        void UpdateSymmetryFlags(const arma::mat &M, SymmetryFlags &flags, bool fullTensorRotation,
                                 double relTol)
        {
            arma::mat A = M;
            if (!fullTensorRotation)
                A = A % arma::eye<arma::mat>(3, 3);

            double maxAbs = 0.0;
            double maxOff = 0.0;
            for (arma::uword r = 0; r < 3; ++r)
            {
                for (arma::uword c = 0; c < 3; ++c)
                {
                    const double v = std::abs(A(r, c));
                    maxAbs = std::max(maxAbs, v);
                    if (r != c)
                        maxOff = std::max(maxOff, v);
                }
            }

            if (!std::isfinite(maxAbs) || maxAbs == 0.0)
                return;

            flags.anyTensor = true;
            if (maxOff > relTol * maxAbs)
            {
                flags.allDiag = false;
                flags.allAxialZ = false;
                flags.allIsotropic = false;
                return;
            }

            const double a = A(0, 0);
            const double b = A(1, 1);
            const double c = A(2, 2);
            const double mean = (a + b + c) / 3.0;
            const double maxDev = std::max({std::abs(a - mean), std::abs(b - mean), std::abs(c - mean)});
            if (maxDev > relTol * maxAbs)
                flags.allIsotropic = false;

            const bool xy_eq = (std::abs(a - b) <= relTol * maxAbs);
            if (!xy_eq && maxDev > relTol * maxAbs)
                flags.allAxialZ = false;
        }
        std::string AutoDetectSopheSymmetry(const SpinAPI::system_ptr &system,
                                            const SpinAPI::interaction_ptr &fieldInteraction,
                                            const std::vector<std::string> &h0list, bool fullTensorRotation)
        {
            if (system == nullptr)
                return "c1";

            const double relTol = 1e-8;
            SymmetryFlags flags;

            for (const auto &name : h0list)
            {
                auto inter = system->interactions_find(name);
                if (inter == nullptr)
                    continue;
                if (!SpinAPI::IsStatic(*inter))
                    continue;

                if (inter->Type() == SpinAPI::InteractionType::SingleSpin)
                {
                    arma::mat R = PassiveZYZRotation(inter->Framelist());
                    arma::mat Rt = R.t();
                    for (const auto &spin : inter->Group1())
                    {
                        arma::mat G = arma::conv_to<arma::mat>::from(spin->GetTensor().LabFrame());
                        if (inter->IgnoreTensors())
                            G = arma::eye<arma::mat>(3, 3);
                        G = Rt * G * Rt.t();
                        UpdateSymmetryFlags(G, flags, fullTensorRotation, relTol);
                    }
                }
                else if (SpinAPI::HasTensor(*inter))
                {
                    arma::mat A = arma::conv_to<arma::mat>::from(inter->CouplingTensor()->LabFrame());
                    arma::mat R = PassiveZYZRotation(inter->Framelist());
                    arma::mat Rt = R.t();
                    A = Rt * A * Rt.t();
                    UpdateSymmetryFlags(A, flags, fullTensorRotation, relTol);
                }
                else if (inter->Type() == SpinAPI::InteractionType::Zfs)
                {
                    const double D = inter->Dvalue();
                    const double E = inter->Evalue();
                    const double maxAbs = std::max(std::abs(D), std::abs(E));
                    if (maxAbs > relTol)
                    {
                        flags.anyTensor = true;
                        flags.allIsotropic = false;
                        if (std::abs(E) > relTol)
                        {
                            flags.allAxialZ = false;
                        }
                    }
                }
                else if (inter->Type() == SpinAPI::InteractionType::SemiClassicalField)
                {
                    return "c1";
                }
            }

            if (fieldInteraction != nullptr)
            {
                arma::mat R = PassiveZYZRotation(fieldInteraction->Framelist());
                arma::mat Rt = R.t();
                for (const auto &spin : fieldInteraction->Group1())
                {
                    arma::mat G = arma::conv_to<arma::mat>::from(spin->GetTensor().LabFrame());
                    if (fieldInteraction->IgnoreTensors())
                        G = arma::eye<arma::mat>(3, 3);
                    G = Rt * G * Rt.t();
                    UpdateSymmetryFlags(G, flags, fullTensorRotation, relTol);
                }
            }

            if (!flags.anyTensor || flags.allIsotropic)
                return "o3";
            if (flags.allAxialZ)
                return "dinfh";
            if (flags.allDiag)
                return "d2h";
            return "c1";
        }
    } // namespace OrientationSampling
} // namespace RunSection::General::Resonance
