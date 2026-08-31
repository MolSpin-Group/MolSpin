/////////////////////////////////////////////////////////////////////////
// ResonanceMagneticMomentBuilder (RunSection::General::Resonance)
// ------------------
// Builds orientation-specific transverse magnetic-dipole operators used by
// General resonance line intensities.
//
// Ownership:
//   * SpinAPI owns spin tensors, interaction frames, Hilbert-space embedding,
//     and interaction prefactors.
//   * The caller owns detection-spin / Zeeman-interaction selection.
//   * This builder composes those inputs with the same molecular-to-lab
//     orientation matrix used by the General Hamiltonian.
//
// For a Zeeman term H = S . g . B, the laboratory transverse operators are
//
//     mu_x = dH/dB_x,    mu_y = dH/dB_y.
//
// The perturbative/exact resonance solvers consume only the resulting matrices;
// they do not parse g tensors or Euler angles.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_Resonance_ResonanceMagneticMomentBuilder
#define MOD_RunSection_General_Resonance_ResonanceMagneticMomentBuilder

#include "Interaction.h"
#include "PowderGrid.h"
#include "Spin.h"
#include "SpinSpace.h"
#include "ResonanceTypes.h"

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <string>
#include <vector>

namespace RunSection::General::Resonance
{
    struct ResonanceMagneticMomentTerm
    {
        SpinAPI::spin_ptr spin;
        SpinAPI::interaction_ptr zeeman;
    };

    class ResonanceMagneticMomentBuilder
    {
    private:
        static bool BuildTerm(
            SpinAPI::SpinSpace &space,
            const ResonanceMagneticMomentTerm &term,
            const arma::mat &moleculeToLab,
            bool fullTensorRotation,
            ResonanceDetectionOperator &channel,
            std::string &error)
        {
            const arma::uword dimension=
                space.HilbertSpaceDimensions();

            if (term.spin==nullptr ||
                term.zeeman==nullptr)
            {
                error =
                    "resonance magnetic-moment term contains a null spin or interaction";
                return false;
            }
            if (!space.Contains(term.spin))
            {
                error =
                    "resonance magnetic-moment detection spin is outside the supplied SpinSpace";
                return false;
            }
            if (term.zeeman->Type()!=
                SpinAPI::InteractionType::SingleSpin)
            {
                error =
                    "resonance magnetic-moment interaction must be a single-spin Zeeman term";
                return false;
            }

            const auto group=term.zeeman->Group1();
            if (std::find(
                    group.begin(),group.end(),
                    term.spin)==group.end())
            {
                error =
                    "resonance magnetic-moment Zeeman interaction does not own the detection spin";
                return false;
            }

            arma::cx_mat Sx,Sy,Sz;
            if (!space.CreateOperator(
                    arma::conv_to<arma::cx_mat>::from(
                        term.spin->Sx()),
                    term.spin,Sx) ||
                !space.CreateOperator(
                    arma::conv_to<arma::cx_mat>::from(
                        term.spin->Sy()),
                    term.spin,Sy) ||
                !space.CreateOperator(
                    arma::conv_to<arma::cx_mat>::from(
                        term.spin->Sz()),
                    term.spin,Sz))
            {
                error =
                    "failed to embed a resonance magnetic-moment spin operator";
                return false;
            }

            arma::mat g=
                term.zeeman->IgnoreTensors()
                ? arma::eye<arma::mat>(3,3)
                : arma::conv_to<arma::mat>::from(
                    term.spin->GetTensor().LabFrame());
            if (g.n_rows!=3 ||
                g.n_cols!=3 ||
                !g.is_finite())
            {
                error =
                    "resonance magnetic-moment spin tensor must be finite 3x3";
                return false;
            }

            const arma::vec frame=
                term.zeeman->Framelist();
            if (frame.n_elem<3 ||
                !frame.is_finite())
            {
                error =
                    "resonance magnetic-moment interaction frame must contain three finite ZYZ angles";
                return false;
            }

            arma::mat frameToMolecule;
            if (!SpinAPI::CreateZYZRotationMatrix(
                    frame(0),frame(1),frame(2),
                    frameToMolecule))
            {
                error =
                    "failed to construct resonance magnetic-moment interaction frame";
                return false;
            }

            g=
                frameToMolecule*g*
                frameToMolecule.t();
            g=
                moleculeToLab*g*
                moleculeToLab.t();

            if (!fullTensorRotation)
                g%=arma::eye<arma::mat>(3,3);

            double prefactor=
                term.zeeman->Prefactor();
            if (term.zeeman->AddCommonPrefactor())
                prefactor*=8.79410005e+1;

            if (!std::isfinite(prefactor))
            {
                error =
                    "resonance magnetic-moment prefactor is non-finite";
                return false;
            }

            channel.x=
                prefactor*
                (g(0,0)*Sx+
                 g(1,0)*Sy+
                 g(2,0)*Sz);
            channel.y=
                prefactor*
                (g(0,1)*Sx+
                 g(1,1)*Sy+
                 g(2,1)*Sz);

            if (channel.x.n_rows!=dimension ||
                channel.x.n_cols!=dimension ||
                channel.y.n_rows!=dimension ||
                channel.y.n_cols!=dimension ||
                !channel.x.is_finite() ||
                !channel.y.is_finite())
            {
                error =
                    "resonance magnetic-moment operator is non-finite or has invalid dimensions";
                return false;
            }
            return true;
        }

    public:
        static bool BuildTransverseChannels(
            SpinAPI::SpinSpace &space,
            const std::vector<ResonanceMagneticMomentTerm> &terms,
            const arma::mat &moleculeToLab,
            bool fullTensorRotation,
            std::vector<ResonanceDetectionOperator> &channels,
            std::string &error)
        {
            error.clear();
            channels.clear();

            if (moleculeToLab.n_rows!=3 ||
                moleculeToLab.n_cols!=3 ||
                !moleculeToLab.is_finite())
            {
                error =
                    "resonance magnetic-moment orientation must be a finite 3x3 matrix";
                return false;
            }
            if (terms.empty())
            {
                error =
                    "resonance magnetic-moment builder requires at least one detection term";
                return false;
            }
            if (space.HilbertSpaceDimensions()==0)
            {
                error =
                    "resonance magnetic-moment spin space is empty";
                return false;
            }

            channels.reserve(terms.size());
            for (const auto &term:terms)
            {
                ResonanceDetectionOperator channel;
                if (!BuildTerm(
                        space,term,moleculeToLab,
                        fullTensorRotation,
                        channel,error))
                    return false;
                channels.push_back(channel);
            }
            return true;
        }

        static bool BuildTransverse(
            SpinAPI::SpinSpace &space,
            const std::vector<ResonanceMagneticMomentTerm> &terms,
            const arma::mat &moleculeToLab,
            bool fullTensorRotation,
            arma::cx_mat &muX,
            arma::cx_mat &muY,
            std::string &error)
        {
            std::vector<ResonanceDetectionOperator>
                channels;
            if (!BuildTransverseChannels(
                    space,terms,moleculeToLab,
                    fullTensorRotation,
                    channels,error))
                return false;

            const arma::uword dimension=
                space.HilbertSpaceDimensions();
            muX.zeros(dimension,dimension);
            muY.zeros(dimension,dimension);

            for (const auto &channel:channels)
            {
                muX+=channel.x;
                muY+=channel.y;
            }

            if (!muX.is_finite() ||
                !muY.is_finite())
            {
                error =
                    "resonance magnetic-moment operators are non-finite";
                return false;
            }
            return true;
        }
    };

}

#endif
