/////////////////////////////////////////////////////////////////////////
// SSNakajimaZwanzigBuilder implementation (RunSection::General::SS)
// ------------------
// Constructs interaction-derived Nakajima-Zwanzig relaxation in the static
// Hamiltonian eigenbasis and returns it in the propagation basis.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// DEVELOPER WORKFLOW / OWNERSHIP MAP
// ----------------------------------------------------------------------
// Interaction-derived Nakajima-Zwanzig relaxation builder.
//
// What is done here:
//   - Diagonalizes the static Hamiltonian and forms Bohr-frequency matrices.
//   - Transforms fluctuating interaction operators to the energy basis.
//   - Combines their correlation expansions with the NZ spectral-density/memory-kernel algebra and transforms the final superoperator back once.
//
// Connections to the General framework / SpinAPI:
//   - Uses SSInteractionRelaxation for operator/correlation preprocessing.
//   - Delegates frequency/spectral-density/relaxation-tensor algebra to SpinAPI::NakajimaZwanzig.
//   - Called from SSLiouvillianBuilder; MultiSS inherits it locally through SS.
//
// Why this ownership is used:
//   - The energy basis makes the free evolution entering the memory integral diagonal in Bohr frequencies.
//   - The completed tensor is transformed back only once to avoid repeated basis changes and convention drift.
//
// Mathematical / physical references:
//   - Nakajima projection formalism: Prog. Theor. Phys. 20, 948-959 (1958), DOI: 10.1143/PTP.20.948; MolSpin reactive-NZ reference: DOI: 10.1063/5.0040519.
//   - Application/validation for g-tensor anisotropy in MolSpin: DOI: 10.1021/jacs.5c06173.
//
// TODO:
//   - Do not interpret the present additive treatment of separate Interaction objects as including cross-interaction correlations; add an explicit shared stochastic-source representation first.
/////////////////////////////////////////////////////////////////////////

#include "SSNakajimaZwanzigBuilder.h"
#include "SSInteractionRelaxation.h"

#include "Interaction.h"
#include "NakajimaZwanzig.h"
#include "ObjectParser.h"
#include "SpinSpace.h"
#include "SpinSystem.h"

#include <vector>

namespace RunSection::General::SS
{
    namespace
    {
        bool Fail(std::string &error, const std::string &message)
        {
            error = message;
            return false;
        }

        bool AddOperatorSet(const std::vector<arma::cx_mat> &labOperators,
            const arma::cx_mat &eigenvectors, const arma::cx_mat &omega,
            const SpinAPI::Relaxation::CorrelationExpansion &correlations,
            arma::cx_mat &total, std::string &error)
        {
            std::vector<arma::cx_mat> operators;
            operators.reserve(labOperators.size());
            for (const auto &op : labOperators)
                operators.push_back(eigenvectors.t() * op * eigenvectors);

            for (std::size_t first = 0; first < operators.size(); ++first)
            {
                const std::size_t secondBegin = correlations.DiagonalOnly() ? first : 0;
                const std::size_t secondEnd = correlations.DiagonalOnly() ? first + 1 : operators.size();
                for (std::size_t second = secondBegin; second < secondEnd; ++second)
                {
                    const std::vector<SpinAPI::Relaxation::ExponentialTerm> *terms = nullptr;
                    if (!correlations.Terms(first,second,terms,&error)) return false;
                    if (terms->empty()) continue;
                    arma::cx_mat spectralDensity, contribution;
                    if (!SpinAPI::NakajimaZwanzig::SpectralDensity(
                            *terms,omega,spectralDensity,&error) ||
                        !SpinAPI::NakajimaZwanzig::RelaxationTensor(
                            operators[first],operators[second],spectralDensity,contribution,&error))
                        return false;
                    total += contribution;
                }
            }
            return true;
        }
    }

    bool SSNakajimaZwanzigBuilder::Build(
        const SpinAPI::system_ptr &system, SpinAPI::SpinSpace &space,
        const arma::sp_cx_mat &hamiltonian, const arma::mat &molecularToLab,
        arma::sp_cx_mat &relaxation,
        std::string &error)
    {
        error.clear();
        relaxation.zeros(hamiltonian.n_rows * hamiltonian.n_rows,
                         hamiltonian.n_rows * hamiltonian.n_rows);
        if (!system) return Fail(error,"cannot build NZ relaxation for a null SpinSystem");
        if (hamiltonian.n_rows == 0 || hamiltonian.n_rows != hamiltonian.n_cols)
            return Fail(error,"NZ relaxation requires a non-empty square static Hamiltonian");

        arma::vec eigenvalues;
        arma::cx_mat eigenvectors;
        if (!arma::eig_sym(eigenvalues,eigenvectors,arma::cx_mat(hamiltonian)))
            return Fail(error,"failed to diagonalize the static Hamiltonian for NZ relaxation");
        arma::cx_mat omega;
        if (!SpinAPI::NakajimaZwanzig::FrequencyMatrix(eigenvalues,omega,&error)) return false;

        // The memory integral is evaluated in the energy eigenbasis, where
        // each matrix element has a definite Bohr frequency. The completed
        // superoperator is transformed back once to the propagation basis.
        arma::cx_mat totalEigenbasis(omega.n_rows,omega.n_cols,arma::fill::zeros);
        bool any = false;
        space.UseSuperoperatorSpace(false);

        for (const auto &interaction : system->Interactions())
        {
            if (!interaction) continue;
            bool relaxationEnabled = true;
            if (interaction->Properties()->Get("relaxation",relaxationEnabled) &&
                !relaxationEnabled) continue;
            bool hasCorrelation = false;
            if (!SSInteractionRelaxation::HasCorrelationInput(interaction,hasCorrelation,error)) return false;
            if (!hasCorrelation) continue;

            int opsMode = 0, termsMode = 0, multiExponential = 0;
            interaction->Properties()->Get("ops",opsMode);
            interaction->Properties()->Get("terms",termsMode);
            interaction->Properties()->Get("def_multexpo",multiExponential);
            if (opsMode != 0 && opsMode != 1)
                return Fail(error,"NZ ops must be 0 (rank-0/2 spherical) or 1 (Cartesian)");
            if (termsMode != 0 && termsMode != 1)
                return Fail(error,"NZ terms must be 0 (cross terms) or 1 (autocorrelation only)");
            if (multiExponential == 1 && opsMode != 1)
                return Fail(error,"def_multexpo=1 correlation matrices require ops=1 Cartesian ordering");

            SpinAPI::Relaxation::CorrelationExpansion correlations;
            const std::size_t operatorCount = opsMode == 1 ? 9 : 6;
            if (!SSInteractionRelaxation::BuildCorrelationExpansion(
                    interaction,operatorCount,termsMode,correlations,error)) return false;
            any = true;

            const auto group1 = interaction->Group1();
            const auto group2 = interaction->Group2();
            if (group1.empty())
                return Fail(error,"NZ Interaction \"" + interaction->Name() + "\" has empty group1");
            if (interaction->Type() == SpinAPI::InteractionType::SingleSpin)
            {
                for (const auto &spin1 : group1)
                {
                    std::vector<arma::cx_mat> operators;
                    if (!SSInteractionRelaxation::BuildOperatorBasis(
                            space,interaction,spin1,nullptr,opsMode,molecularToLab,operators,error) ||
                        !AddOperatorSet(operators,eigenvectors,omega,correlations,totalEigenbasis,error))
                        return false;
                }
            }
            else if (interaction->Type() == SpinAPI::InteractionType::DoubleSpin)
            {
                if (group2.empty())
                    return Fail(error,"NZ double-spin Interaction \"" + interaction->Name() + "\" has empty group2");
                for (const auto &spin1 : group1)
                    for (const auto &spin2 : group2)
                    {
                        std::vector<arma::cx_mat> operators;
                        if (!SSInteractionRelaxation::BuildOperatorBasis(
                                space,interaction,spin1,spin2,opsMode,molecularToLab,operators,error) ||
                            !AddOperatorSet(operators,eigenvectors,omega,correlations,totalEigenbasis,error))
                            return false;
                    }
            }
            else return Fail(error,"NZ relaxation only supports single- and double-spin Interactions");
        }

        // MultiSS may contain manifolds without a relaxation-enabled
        // Interaction. Their local contribution is exactly zero.
        if (!any) return true;
        if (!totalEigenbasis.is_finite())
            return Fail(error,"NZ relaxation contains non-finite entries");
        if (!space.TransformSuperoperatorFromEigenbasis(
                eigenvectors,arma::sp_cx_mat(totalEigenbasis),relaxation))
            return Fail(error,"failed to transform NZ relaxation from the Hamiltonian eigenbasis to the propagation basis");
        return true;
    }
}
