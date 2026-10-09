/////////////////////////////////////////////////////////////////////////
// SSRedfieldBuilder implementation (RunSection::General::SS)
// ------------------
// Constructs interaction-derived Redfield relaxation in the static
// Hamiltonian eigenbasis and returns it in the propagation basis.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// DEVELOPER WORKFLOW / OWNERSHIP MAP
// ----------------------------------------------------------------------
// Interaction-derived Redfield relaxation builder.
//
// What is done here:
//   - Diagonalizes the static Hamiltonian and constructs transition-frequency matrices.
//   - Transforms fluctuating interaction operators to the energy basis.
//   - Builds the selected Redfield spectral density/tensor and transforms the resulting superoperator back to the propagation basis.
//
// Connections to the General framework / SpinAPI:
//   - Uses SSInteractionRelaxation for shared operator/correlation preprocessing.
//   - Delegates spectral-density and Redfield tensor algebra to SpinAPI::Redfield.
//   - Called from SSLiouvillianBuilder and therefore inherited locally by MultiSS.
//
// Why this ownership is used:
//   - Redfield is kept distinct from NZ because the two approximations use different generator constructions even when they share the same fluctuation input.
//   - Initial-state slippage remains explicitly unsupported in General rather than being silently omitted.
//
// Mathematical / physical references:
//   - Redfield weak-coupling relaxation framework; IBM J. Res. Dev. 1, 19-31 (1957), DOI: 10.1147/rd.11.0019.
//
// TODO:
//   - General-framework slippage and cross-Interaction correlations remain missing parity items relative to dedicated legacy Redfield implementations.
/////////////////////////////////////////////////////////////////////////

#include "SSRedfieldBuilder.h"
#include "SSInteractionRelaxation.h"

#include "Interaction.h"
#include "ObjectParser.h"
#include "Redfield.h"
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
            const arma::cx_mat &eigenvectors, const arma::cx_mat &frequencies,
            const SpinAPI::Relaxation::CorrelationExpansion &correlations,
            SpinAPI::Relaxation::SpectralDensityFunction spectralFunction,
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
                    if (!SpinAPI::Redfield::SpectralDensity(
                            *terms,frequencies,spectralFunction,spectralDensity,&error) ||
                        !SpinAPI::Redfield::RelaxationTensor(
                            operators[first],operators[second],spectralDensity,contribution,&error))
                        return false;
                    total += contribution;
                }
            }
            return true;
        }
    }

    bool SSRedfieldBuilder::Build(const SpinAPI::system_ptr &system,
        SpinAPI::SpinSpace &space, const arma::sp_cx_mat &hamiltonian,
        const arma::mat &molecularToLab,
        arma::sp_cx_mat &relaxation, std::string &error)
    {
        error.clear();
        const arma::uword dimension = hamiltonian.n_rows;
        relaxation.zeros(dimension * dimension, dimension * dimension);
        if (!system) return Fail(error,"cannot build Redfield relaxation for a null SpinSystem");
        if (dimension == 0 || hamiltonian.n_cols != dimension)
            return Fail(error,"Redfield relaxation requires a non-empty square static Hamiltonian");

        arma::vec eigenvalues;
        arma::cx_mat eigenvectors;
        if (!arma::eig_sym(eigenvalues,eigenvectors,arma::cx_mat(hamiltonian)))
            return Fail(error,"failed to diagonalize the static Hamiltonian for Redfield relaxation");
        arma::cx_mat frequencies;
        if (!SpinAPI::Redfield::FrequencyMatrix(eigenvalues,frequencies,&error)) return false;

        arma::cx_mat totalEigenbasis(dimension * dimension,dimension * dimension,arma::fill::zeros);
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
            int spectralDensityMode = 0, slippage = 0;
            interaction->Properties()->Get("ops",opsMode);
            interaction->Properties()->Get("terms",termsMode);
            interaction->Properties()->Get("def_multexpo",multiExponential);
            interaction->Properties()->Get("def_specdens",spectralDensityMode);
            interaction->Properties()->Get("slip",slippage);
            if (opsMode != 0 && opsMode != 1)
                return Fail(error,"Redfield ops must be 0 (rank-0/2 spherical) or 1 (Cartesian)");
            if (termsMode != 0 && termsMode != 1)
                return Fail(error,"Redfield terms must be 0 (cross terms) or 1 (autocorrelation only)");
            if (spectralDensityMode != 0 && spectralDensityMode != 1)
                return Fail(error,"Redfield def_specdens must be 0 or 1");
            if (multiExponential == 1 && opsMode != 1)
                return Fail(error,"def_multexpo=1 correlation matrices require ops=1 Cartesian ordering");
            if (slippage != 0)
                return Fail(error,"Redfield initial-state slippage is not available in the General framework; use slip=0 or a dedicated Redfield task");

            SpinAPI::Relaxation::CorrelationExpansion correlations;
            const std::size_t operatorCount = opsMode == 1 ? 9 : 6;
            if (!SSInteractionRelaxation::BuildCorrelationExpansion(
                    interaction,operatorCount,termsMode,correlations,error)) return false;
            const auto spectralFunction = spectralDensityMode == 1
                ? SpinAPI::Relaxation::SpectralDensityFunction::RealLorentzian
                : SpinAPI::Relaxation::SpectralDensityFunction::ComplexOneSided;
            any = true;

            const auto group1 = interaction->Group1();
            const auto group2 = interaction->Group2();
            if (group1.empty())
                return Fail(error,"Redfield Interaction \"" + interaction->Name() + "\" has empty group1");
            if (interaction->Type() == SpinAPI::InteractionType::SingleSpin)
            {
                for (const auto &spin1 : group1)
                {
                    std::vector<arma::cx_mat> operators;
                    if (!SSInteractionRelaxation::BuildOperatorBasis(
                            space,interaction,spin1,nullptr,opsMode,molecularToLab,operators,error) ||
                        !AddOperatorSet(operators,eigenvectors,frequencies,correlations,
                            spectralFunction,totalEigenbasis,error)) return false;
                }
            }
            else if (interaction->Type() == SpinAPI::InteractionType::DoubleSpin)
            {
                if (group2.empty())
                    return Fail(error,"Redfield double-spin Interaction \"" + interaction->Name() + "\" has empty group2");
                for (const auto &spin1 : group1)
                    for (const auto &spin2 : group2)
                    {
                        std::vector<arma::cx_mat> operators;
                        if (!SSInteractionRelaxation::BuildOperatorBasis(
                                space,interaction,spin1,spin2,opsMode,molecularToLab,operators,error) ||
                            !AddOperatorSet(operators,eigenvectors,frequencies,correlations,
                                spectralFunction,totalEigenbasis,error)) return false;
                    }
            }
            else return Fail(error,"Redfield relaxation only supports single- and double-spin Interactions");
        }

        if (!any) return true;
        if (!totalEigenbasis.is_finite())
            return Fail(error,"Redfield relaxation contains non-finite entries");
        if (!space.TransformSuperoperatorFromEigenbasis(
                eigenvectors,arma::sp_cx_mat(totalEigenbasis),relaxation))
            return Fail(error,"failed to transform Redfield relaxation from the Hamiltonian eigenbasis to the propagation basis");
        return true;
    }
}
