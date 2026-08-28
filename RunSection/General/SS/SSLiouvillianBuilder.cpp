/////////////////////////////////////////////////////////////////////////
// SSLiouvillianBuilder implementation (RunSection::General::SS)
// ----------------------------------------------------------------------
// HIERARCHY AND OWNERSHIP
//   SpinAPI primitives
//       -> General/SS::SSLiouvillianBuilder                [this file]
//       -> General/MultiSS::MultiSSSystemPreparation
//       -> MultiSSNetworkBuilder
//       -> TaskMultiSSGeneral.
//
//   This layer owns *one-manifold superspace physics*: Hamiltonian construction,
//   initial density preparation, and relaxation Operators local to that
//   SpinSystem.  It deliberately DOES NOT add Transition reaction loss or
//   inter-system creation terms.  Those are graph edges and are compiled once
//   by General/MultiSS.  This separation is the key protection against double
//   counting kinetic channels.
//
// LOCAL LIOUVILLIAN
//       L_internal rho = -i[H,rho] + R_local rho.
//
//   Existing SpinAPI relaxation Operator implementations are reused.  The
//   established NZ algebra is extracted separately into SpinAPI so
//   published behavior (DOI: 10.1021/jacs.5c06173) can be parity-tested before
//   it is wired into this production builder.  The formal reactive NZ reference
//   is DOI: 10.1063/5.0040519; it is not silently substituted here.
//
// ORIENTATION CONTRACT
//   The same molecular-to-laboratory rotation supplied by General/MultiSS is
//   used for every rigidly related manifold.  `rotated_zyz` keeps the full
//   Hamiltonian; `rotated_sa` selects the existing high-field/secular SpinAPI
//   construction.  General/MultiSS never samples separate random orientations
//   for separate electronic manifolds of one molecular network.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// DEVELOPER WORKFLOW / OWNERSHIP MAP
// ----------------------------------------------------------------------
// One-manifold SS Hamiltonian, initial density and internal Liouvillian builder.
//
// What is done here:
//   - Builds the local Hamiltonian for the current crystallite.
//   - Prepares/rotates/normalizes the local initial density.
//   - Constructs -i[H,rho] and adds local explicit Operator relaxation plus optional interaction-derived NZ or Redfield terms.
//
// Connections to the General framework / SpinAPI:
//   - SpinAPI::SpinSpace owns Hamiltonians, rotations, state projectors and explicit relaxation operators.
//   - SSSystemPreparation adds terminal one-system reaction loss.
//   - MultiSSSystemPreparation reuses this builder for every local manifold but adds network edges only later.
//
// Why this ownership is used:
//   - Local relaxation is separated from Transition kinetics so MultiSS cannot double-count a channel as both a dissipator and a graph edge.
//   - All relaxation contributions are returned in one propagation basis before addition.
//
// Mathematical / physical references:
//   - Nakajima projection formalism: Prog. Theor. Phys. 20, 948-959 (1958), DOI: 10.1143/PTP.20.948; MolSpin reactive-NZ reference: DOI: 10.1063/5.0040519.
//   - Redfield weak-coupling relaxation framework; IBM J. Res. Dev. 1, 19-31 (1957), DOI: 10.1147/rd.11.0019.
/////////////////////////////////////////////////////////////////////////

#include "SSLiouvillianBuilder.h"
#include "SSNakajimaZwanzigBuilder.h"
#include "SSRedfieldBuilder.h"

#include "ObjectParser.h"
#include "Operator.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "State.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace RunSection::General::SS
{
    namespace
    {
        bool IsIdentityRotation(const arma::mat &R)
        {
            if (R.n_rows != 3 || R.n_cols != 3) return false;
            return arma::norm(R - arma::eye<arma::mat>(3, 3), "fro") < 1.0e-13;
        }
    }

    bool SSLiouvillianBuilder::BuildHamiltonian(const SpinAPI::system_ptr &system,
        SpinAPI::SpinSpace &space, SSHamiltonianMode mode, const arma::mat &rotation,
        arma::sp_cx_mat &hamiltonian, std::string &error)
    {
        error.clear();
        if (system == nullptr) { error = "cannot build a superspace Hamiltonian for a null SpinSystem"; return false; }
        space.UseSuperoperatorSpace(false);
        bool ok = false;
        if (mode == SSHamiltonianMode::FixedFull)
            ok = space.StaticHamiltonian(hamiltonian);
        else if (mode == SSHamiltonianMode::RotatedFull)
            ok = space.StaticHamiltonianRotatedZYZ(rotation, hamiltonian);
        else
            ok = space.StaticHamiltonianRotatedSA(rotation, hamiltonian);
        if (!ok)
        {
            error = "failed to construct the local Hamiltonian for SpinSystem \"" + system->Name() + "\"";
            return false;
        }
        const arma::cx_mat dense(hamiltonian);
        const double scale = std::max(1.0, arma::norm(dense, "fro"));
        if (arma::norm(dense - dense.t(), "fro") > 1.0e-10 * scale)
        {
            error = "local Hamiltonian for SpinSystem \"" + system->Name() + "\" is not Hermitian";
            return false;
        }
        return true;
    }

    bool SSLiouvillianBuilder::BuildInitialDensity(const SpinAPI::system_ptr &system,
        SpinAPI::SpinSpace &space, const arma::sp_cx_mat &hamiltonian,
        const arma::mat &rotation,
        arma::cx_mat &density, std::string &error)
    {
        error.clear(); density.reset();
        if (system == nullptr) { error = "cannot prepare an initial density for a null SpinSystem"; return false; }

        const auto initialStates = system->InitialState();
        const arma::uword d = space.HilbertSpaceDimensions();
        density.zeros(d, d);
        if (initialStates.empty())
        {
            // In a reaction network an unprepared manifold is a legitimate
            // zero-population destination.  This differs from single-system
            // HSGeneral, where absence of an initial state is usually an error.
            return true;
        }

        if (system->InitialStateFrame() == SpinAPI::StateFrame::Eigen &&
            (initialStates.size() != 1 || initialStates.front() != nullptr))
        {
            error = "initial-state frame=eigen for SpinSystem \"" + system->Name() +
                "\" requires exactly one Thermal initial state";
            return false;
        }

        std::vector<double> weights = system->Weights();
        if (weights.empty()) weights.assign(initialStates.size(), 1.0);
        if (weights.size() != initialStates.size())
        { error = "initial-state weight count does not match initial states in \"" + system->Name() + "\""; return false; }
        double sum = 0.0;
        for (double w : weights)
        {
            if (!std::isfinite(w) || w < 0.0) { error = "initial-state weights must be finite and non-negative"; return false; }
            sum += w;
        }
        if (!(sum > 0.0)) { error = "initial-state weights sum to zero"; return false; }
        for (double &w : weights) w /= sum;

        for (size_t i = 0; i < initialStates.size(); ++i)
        {
            arma::cx_mat component;
            if (initialStates[i] == nullptr)
            {
                const SpinAPI::StateFrame frame = system->InitialStateFrame();
                const std::vector<std::string> thermalList = system->ThermalHamiltonianList();
                if (frame == SpinAPI::StateFrame::Eigen)
                {
                    // The canonical density is defined by the interactions in
                    // `thermalhamiltonian`, not by every term retained in the
                    // propagation generator.  Rebuild that selected Hamiltonian
                    // with the full crystallite rotation.  In particular, a
                    // high-field/secular propagation approximation must not
                    // silently change rho_eq = exp(-hbar H_th/k_B T)/Z.
                    arma::sp_cx_mat thermalHamiltonian;
                    if (!space.BaseHamiltonianRotatedZYZ(thermalList, rotation,
                            thermalHamiltonian) ||
                        !space.ThermalStateFromHamiltonian(arma::cx_mat(thermalHamiltonian),
                            system->Temperature(), component))
                    {
                        error = "failed to construct orientation-specific thermal initial density for \"" +
                            system->Name() + "\"";
                        return false;
                    }
                }
                else
                {
                    if (!space.GetThermalState(space, system->Temperature(), thermalList, component))
                    {
                        error = "failed to construct thermal initial density for \"" +
                            system->Name() + "\"";
                        return false;
                    }
                    if (frame == SpinAPI::StateFrame::Molecular && !IsIdentityRotation(rotation))
                    {
                        arma::cx_mat rotated;
                        if (!space.RotateState(component, rotation, rotated))
                        {
                            error = "failed to rotate molecular-frame thermal initial density for \"" +
                                system->Name() + "\"";
                            return false;
                        }
                        component = std::move(rotated);
                    }
                }
            }
            else
            {
                if (!space.GetState(initialStates[i], component))
                { error = "failed to construct initial State \"" + initialStates[i]->Name() + "\""; return false; }
                if (system->InitialStateFrame() == SpinAPI::StateFrame::Molecular &&
                    !IsIdentityRotation(rotation))
                {
                    arma::cx_mat rotated;
                    if (!space.RotateState(component, rotation, rotated))
                    { error = "failed to rotate molecular-frame initial State \"" + initialStates[i]->Name() + "\""; return false; }
                    component = std::move(rotated);
                }
            }
            const arma::cx_double tr = arma::trace(component);
            if (!std::isfinite(std::real(tr)) || !std::isfinite(std::imag(tr)) ||
                std::abs(std::imag(tr)) > 1.0e-10 * std::max(1.0, std::abs(std::real(tr))) ||
                !(std::real(tr) > 0.0))
            { error = "initial-state component has invalid trace"; return false; }
            density += weights[i] * component / std::real(tr);
        }

        if (system->InitialStateCoherences() == SpinAPI::InitialStateCoherenceMode::DephaseEigenbasis)
        {
            arma::cx_mat dephased;
            if (!space.DephaseStateInEigenbasis(density, arma::cx_mat(hamiltonian), dephased))
            { error = "failed to dephase initial density in local Hamiltonian eigenbasis"; return false; }
            density = std::move(dephased);
        }

        double population = 1.0;
        if (system->GetProperties() != nullptr)
            system->GetProperties()->Get("initialpopulation", population);
        if (!std::isfinite(population) || population < 0.0)
        { error = "initialpopulation must be finite and non-negative for \"" + system->Name() + "\""; return false; }
        density *= population;
        density = 0.5 * (density + density.t());
        return true;
    }

    bool SSLiouvillianBuilder::BuildInternalLiouvillian(const SpinAPI::system_ptr &system,
        SpinAPI::SpinSpace &space, const arma::sp_cx_mat &hamiltonian,
        const arma::mat &rotation, arma::sp_cx_mat &liouvillian,
        std::string &error, SSRelaxationModel relaxationModel)
    {
        error.clear();
        const arma::uword d = hamiltonian.n_rows;
        if (d == 0 || hamiltonian.n_cols != d) { error = "invalid local Hamiltonian dimensions"; return false; }

        space.UseSuperoperatorSpace(true);
        arma::sp_cx_mat left, right;
        if (!space.SuperoperatorFromLeftOperator(hamiltonian, left) ||
            !space.SuperoperatorFromRightOperator(hamiltonian, right))
        { error = "failed to lift local Hamiltonian into superspace"; return false; }
        liouvillian = arma::cx_double(0.0, -1.0) * (left - right);

        // Existing SpinAPI relaxation Operator implementations are reused here.
        // They are local one-manifold physics and therefore belong below the
        // MultiSS network layer. Explicit Operators and the selected
        // interaction-derived model are additive, matching the legacy NZ and
        // Redfield tasks. Every contribution is returned in this Liouvillian's
        // propagation basis before it is added.
        if (!system->Operators().empty())
        {
            arma::vec eigenvalues;
            arma::cx_mat eigenvectors;
            if (!arma::eig_sym(eigenvalues, eigenvectors, arma::cx_mat(hamiltonian)))
            { error = "failed to diagonalize local Hamiltonian for relaxation"; return false; }
            for (const auto &op : system->Operators())
            {
                arma::sp_cx_mat relaxation;
                if (!space.PowderRelaxationOperator(op, eigenvectors, rotation, relaxation))
                { error = "failed to construct relaxation Operator \"" + op->Name() + "\""; return false; }
                liouvillian += relaxation;
            }
        }

        if (relaxationModel == SSRelaxationModel::NakajimaZwanzig)
        {
            arma::sp_cx_mat nz;
            if (!SSNakajimaZwanzigBuilder::Build(system, space, hamiltonian, rotation, nz, error))
                return false;
            liouvillian += nz;
        }
        else if (relaxationModel == SSRelaxationModel::Redfield)
        {
            arma::sp_cx_mat redfield;
            if (!SSRedfieldBuilder::Build(system, space, hamiltonian, rotation, redfield, error))
                return false;
            liouvillian += redfield;
        }
        return true;
    }

    bool SSLiouvillianBuilder::Prepare(const SpinAPI::system_ptr &system,
        SSHamiltonianMode mode, const arma::mat &rotation,
        SSPreparedSystem &prepared, std::string &error,
        SSRelaxationModel relaxationModel)
    {
        prepared = SSPreparedSystem(); error.clear();
        if (system == nullptr) { error = "cannot prepare null SpinSystem"; return false; }
        prepared.system = system;
        prepared.space = std::make_shared<SpinAPI::SpinSpace>(system);
        if (!BuildHamiltonian(system, *prepared.space, mode, rotation, prepared.hamiltonian, error) ||
            !BuildInitialDensity(system, *prepared.space, prepared.hamiltonian, rotation, prepared.initialDensity, error) ||
            !BuildInternalLiouvillian(system, *prepared.space, prepared.hamiltonian, rotation, prepared.internalLiouvillian, error, relaxationModel))
            return false;
        return true;
    }
}
