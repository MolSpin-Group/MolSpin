/////////////////////////////////////////////////////////////////////////
// SSInteractionRelaxation implementation (RunSection::General::SS)
// ------------------
// Input compatibility:
//   def_multexpo=1 : matrix rows are correlation channels, columns are
//                    exponential terms in that channel;
//   def_g=1        : one coupling factor per operator and one tau_c;
//   otherwise      : flat g/tau_c lists define one shared expansion.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "SSInteractionRelaxation.h"

#include "Interaction.h"
#include "ObjectParser.h"
#include "PowderGrid.h"
#include "Spin.h"
#include "SpinSpace.h"
#include "Tensor.h"

#include <cmath>

namespace RunSection::General::SS
{
    namespace
    {
        bool Fail(std::string &error, const std::string &message)
        {
            error = message;
            return false;
        }

        std::string Prefix(const SpinAPI::interaction_ptr &interaction)
        {
            return "interaction-derived relaxation Interaction \"" + interaction->Name() + "\" ";
        }

        bool InteractionToLab(const SpinAPI::interaction_ptr &interaction,
            const arma::mat &molecularToLab, arma::mat &interactionToLab,
            std::string &error)
        {
            if (molecularToLab.n_rows != 3 || molecularToLab.n_cols != 3 ||
                !molecularToLab.is_finite())
                return Fail(error, "relaxation orientation must be a finite 3x3 rotation matrix");

            const arma::vec frame = interaction->Framelist();
            const double alpha = frame.n_elem > 0 ? frame(0) : 0.0;
            const double beta = frame.n_elem > 1 ? frame(1) : 0.0;
            const double gamma = frame.n_elem > 2 ? frame(2) : 0.0;
            arma::mat interactionFrame;
            if (!SpinAPI::CreateZYZRotationMatrix(alpha,beta,gamma,interactionFrame))
                return Fail(error, "failed to construct the Euler frame for Interaction \"" +
                    interaction->Name() + "\"");

            // Correlation channels are supplied in the interaction frame.
            // The crystallite rotation then maps that frame into the lab.
            interactionToLab = molecularToLab * interactionFrame;
            return true;
        }

        bool CreateSpinVector(SpinAPI::SpinSpace &space,
            const SpinAPI::spin_ptr &spin, std::vector<arma::cx_mat> &operators,
            std::string &error)
        {
            operators.resize(3);
            if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sx()), spin, operators[0]) ||
                !space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sy()), spin, operators[1]) ||
                !space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sz()), spin, operators[2]))
                return Fail(error, "failed to create Cartesian relaxation operators for spin \"" + spin->Name() + "\"");
            return true;
        }

        std::vector<arma::cx_mat> RotateSpinVector(
            const std::vector<arma::cx_mat> &labOperators,
            const arma::mat &interactionToLab)
        {
            std::vector<arma::cx_mat> rotated(3);
            for (arma::uword interactionAxis = 0; interactionAxis < 3; ++interactionAxis)
            {
                rotated[interactionAxis].zeros(arma::size(labOperators[0]));
                for (arma::uword labAxis = 0; labAxis < 3; ++labAxis)
                    rotated[interactionAxis] += interactionToLab(labAxis,interactionAxis) *
                        labOperators[labAxis];
            }
            return rotated;
        }

        bool CartesianOperators(SpinAPI::SpinSpace &space,
            const SpinAPI::spin_ptr &spin1, const SpinAPI::spin_ptr &spin2,
            const arma::mat &interactionToLab,
            std::vector<arma::cx_mat> &operators, std::string &error)
        {
            std::vector<arma::cx_mat> spin1Lab;
            if (!CreateSpinVector(space,spin1,spin1Lab,error)) return false;
            const auto spin1Interaction = RotateSpinVector(spin1Lab,interactionToLab);

            if (!spin2)
            {
                // The nine fitted channels are ordered xx,xy,xz,yx,...,zz.
                // For one-spin interactions the first label selects Sx/Sy/Sz;
                // the second labels a spatial fluctuation already contained in
                // the fitted correlation amplitude. It must not introduce an
                // additional field factor here.
                operators = {spin1Interaction[0],spin1Interaction[0],spin1Interaction[0],
                             spin1Interaction[1],spin1Interaction[1],spin1Interaction[1],
                             spin1Interaction[2],spin1Interaction[2],spin1Interaction[2]};
                return true;
            }

            std::vector<arma::cx_mat> spin2Lab;
            if (!CreateSpinVector(space,spin2,spin2Lab,error)) return false;
            const auto spin2Interaction = RotateSpinVector(spin2Lab,interactionToLab);
            operators = {spin1Interaction[0]*spin2Interaction[0],spin1Interaction[0]*spin2Interaction[1],spin1Interaction[0]*spin2Interaction[2],
                         spin1Interaction[1]*spin2Interaction[0],spin1Interaction[1]*spin2Interaction[1],spin1Interaction[1]*spin2Interaction[2],
                         spin1Interaction[2]*spin2Interaction[0],spin1Interaction[2]*spin2Interaction[1],spin1Interaction[2]*spin2Interaction[2]};
            return true;
        }

        bool SphericalOperators(SpinAPI::SpinSpace &space,
            const SpinAPI::interaction_ptr &interaction,
            const SpinAPI::spin_ptr &spin1, const SpinAPI::spin_ptr &spin2,
            const arma::mat &interactionToLab,
            std::vector<arma::cx_mat> &operators, std::string &error)
        {
            std::vector<arma::cx_mat> spin1Lab;
            if (!CreateSpinVector(space,spin1,spin1Lab,error)) return false;
            const auto first = RotateSpinVector(spin1Lab,interactionToLab);
            const arma::cx_double im(0.0,1.0);
            arma::cx_mat t00, t20, tm1, tp1, tm2, tp2;
            if (!spin2)
            {
                const arma::vec fieldLab = interaction->Field();
                if (fieldLab.n_elem != 3)
                    return Fail(error, "single-spin spherical relaxation requires a three-component field");
                // `first` is expressed along the interaction-frame axes:
                //     S_int = R^T S_lab.
                // The spatial field components entering the same spherical
                // tensor must be expressed in that same frame:
                //     B_int = R^T B_lab.
                // Rotating only S while leaving B in the lab frame violates
                // even the rank-0 scalar-product invariant.
                const arma::cx_vec field = arma::conv_to<arma::cx_vec>::from(
                    interactionToLab.t() * fieldLab);
                const auto &sx=first[0], &sy=first[1], &sz=first[2];
                const arma::cx_mat scalar=sx*field(0)+sy*field(1)+sz*field(2);
                t00=scalar/std::sqrt(3.0);
                t20=(3.0*sz*field(2)-scalar)/std::sqrt(6.0);
                tp1=-0.5*(sx*field(2)+sz*field(0)+im*(sy*field(2)+sz*field(1)));
                tm1= 0.5*(sx*field(2)+sz*field(0)-im*(sy*field(2)+sz*field(1)));
                tp2= 0.5*(sx*field(0)-sy*field(1)+im*(sx*field(1)+sy*field(0)));
                tm2= 0.5*(sx*field(0)-sy*field(1)-im*(sx*field(1)+sy*field(0)));
            }
            else
            {
                std::vector<arma::cx_mat> spin2Lab;
                if (!CreateSpinVector(space,spin2,spin2Lab,error)) return false;
                const auto second=RotateSpinVector(spin2Lab,interactionToLab);
                const auto &s1x=first[0], &s1y=first[1], &s1z=first[2];
                const auto &s2x=second[0], &s2y=second[1], &s2z=second[2];
                const arma::cx_mat scalar=s1x*s2x+s1y*s2y+s1z*s2z;
                t00=scalar/std::sqrt(3.0);
                t20=(3.0*s1z*s2z-scalar)/std::sqrt(6.0);
                tp1=-0.5*(s1x*s2z+s1z*s2x+im*(s1y*s2z+s1z*s2y));
                tm1= 0.5*(s1x*s2z+s1z*s2x-im*(s1y*s2z+s1z*s2y));
                tp2= 0.5*(s1x*s2x-s1y*s2y+im*(s1x*s2y+s1y*s2x));
                tm2= 0.5*(s1x*s2x-s1y*s2y-im*(s1x*s2y+s1y*s2x));
            }
            // MolSpin stores the spherical channels as
            // {T00,T20,-T2-1,-T2+1,T2-2,T2+2}. Keep this phase convention
            // before applying optional spatial coefficients.
            operators = {t00,t20,-tm1,-tp1,tm2,tp2};

            int coefficientMode = 0;
            SpinAPI::Tensor parsed(0);
            if (interaction->Properties()->Get("tensor",parsed) &&
                interaction->Properties()->Get("coeff",coefficientMode) && coefficientMode == 1)
            {
                const auto tensor = interaction->CouplingTensor();
                if (!tensor) return Fail(error, "coeff=1 requires a coupling tensor for Interaction \"" + interaction->Name() + "\"");
                const arma::cx_mat A = arma::conv_to<arma::cx_mat>::from(tensor->LabFrame());
                arma::cx_vec am(6,arma::fill::zeros);
                am(0)=(A(0,0)+A(1,1)+A(2,2))/std::sqrt(3.0);
                am(1)=(3.0*A(2,2)-(A(0,0)+A(1,1)+A(2,2)))/std::sqrt(6.0);
                am(2)=0.5*(A(0,2)+A(2,0)+arma::cx_double(0.0,1.0)*(A(1,2)+A(2,1)));
                am(3)=-0.5*(A(0,2)+A(2,0)-arma::cx_double(0.0,1.0)*(A(1,2)+A(2,1)));
                am(4)=0.5*(A(0,0)-A(1,1)-arma::cx_double(0.0,1.0)*(A(0,1)+A(1,0)));
                am(5)=0.5*(A(0,0)-A(1,1)+arma::cx_double(0.0,1.0)*(A(0,1)+A(1,0)));
                operators[0]*=am(0); operators[1]*=am(1); operators[2]*=am(3);
                operators[3]*=am(2); operators[4]*=am(4); operators[5]*=am(5);
            }
            return true;
        }
    }

    bool SSInteractionRelaxation::HasCorrelationInput(
        const SpinAPI::interaction_ptr &interaction, bool &hasInput,
        std::string &error)
    {
        error.clear(); hasInput = false;
        if (!interaction || !interaction->Properties())
            return Fail(error, "cannot inspect correlation input on a null Interaction");
        std::string rawG, rawTau;
        const bool hasG = interaction->Properties()->Get("g",rawG);
        const bool hasTau = interaction->Properties()->Get("tau_c",rawTau);
        if (hasG != hasTau)
            return Fail(error, Prefix(interaction) + "must specify both g and tau_c");
        hasInput = hasG;
        return true;
    }

    bool SSInteractionRelaxation::BuildCorrelationExpansion(
        const SpinAPI::interaction_ptr &interaction, std::size_t operatorCount,
        int terms, SpinAPI::Relaxation::CorrelationExpansion &expansion,
        std::string &error)
    {
        error.clear();
        if (!interaction || !interaction->Properties())
            return Fail(error, "cannot build correlations for a null Interaction");
        if (terms != 0 && terms != 1)
            return Fail(error, Prefix(interaction) + "terms must be 0 (cross terms) or 1 (autocorrelation only)");
        const bool diagonalOnly = terms == 1;

        int multiExponential = 0;
        interaction->Properties()->Get("def_multexpo",multiExponential);
        if (multiExponential != 0 && multiExponential != 1)
            return Fail(error, Prefix(interaction) + "def_multexpo must be 0 or 1");
        if (multiExponential == 1)
        {
            arma::mat amplitudes, tauC;
            if (!interaction->Properties()->GetMatrix("g",amplitudes) ||
                !interaction->Properties()->GetMatrix("tau_c",tauC))
                return Fail(error, Prefix(interaction) + "requires matrix-valued g and tau_c when def_multexpo=1");
            std::string detail;
            if (!SpinAPI::Relaxation::CorrelationExpansion::PerChannel(
                operatorCount,diagonalOnly,amplitudes,tauC,expansion,&detail))
                return Fail(error, Prefix(interaction) + detail);
            return true;
        }

        std::vector<double> amplitudes, tauC;
        if (!interaction->Properties()->GetList("g",amplitudes) ||
            !interaction->Properties()->GetList("tau_c",tauC))
            return Fail(error, Prefix(interaction) + "could not parse g and tau_c lists");
        int definedFactors = 0;
        interaction->Properties()->Get("def_g",definedFactors);
        if (definedFactors != 0 && definedFactors != 1)
            return Fail(error, Prefix(interaction) + "def_g must be 0 or 1");
        std::string detail;
        if (definedFactors == 1)
        {
            if (tauC.size() != 1)
                return Fail(error, Prefix(interaction) + "def_g=1 requires exactly one tau_c");
            if (!SpinAPI::Relaxation::CorrelationExpansion::FromOperatorFactors(
                operatorCount,diagonalOnly,amplitudes,tauC[0],expansion,&detail))
                return Fail(error, Prefix(interaction) + detail);
        }
        else if (!SpinAPI::Relaxation::CorrelationExpansion::Shared(
            operatorCount,diagonalOnly,amplitudes,tauC,expansion,&detail))
            return Fail(error, Prefix(interaction) + detail);
        return true;
    }

    bool SSInteractionRelaxation::BuildOperatorBasis(SpinAPI::SpinSpace &space,
        const SpinAPI::interaction_ptr &interaction,
        const SpinAPI::spin_ptr &spin1, const SpinAPI::spin_ptr &spin2,
        int opsMode, const arma::mat &molecularToLab,
        std::vector<arma::cx_mat> &operators,
        std::string &error)
    {
        error.clear(); operators.clear();
        if (!interaction || !spin1)
            return Fail(error, "cannot build a relaxation operator basis from null objects");
        arma::mat interactionToLab;
        if (!InteractionToLab(interaction,molecularToLab,interactionToLab,error)) return false;
        if (opsMode == 1) return CartesianOperators(space,spin1,spin2,interactionToLab,operators,error);
        if (opsMode == 0) return SphericalOperators(space,interaction,spin1,spin2,interactionToLab,operators,error);
        return Fail(error, "relaxation ops must be 0 (rank-0/2 spherical) or 1 (Cartesian)");
    }
}
