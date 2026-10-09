/////////////////////////////////////////////////////////////////////////
// NuclearZeeman (SpinAPI)
// ------------------
// Isotope-aware nuclear Zeeman constants, construction and validation.
//
// Sign/unit convention:
//   H_Z,n = - g_N (mu_N / hbar) B . I
//
// MolSpin Hamiltonians are expressed in angular-frequency units (rad/ns).
// Therefore the explicit single-spin interaction prefactor is negative for
// nuclei with positive g_N and commonprefactor must be false.
//
// Initial authenticated registry scope: 51V only. Unsupported isotopes fail
// closed. The registry is intentionally extensible after isotope data are
// independently audited.
//
// 51V nuclear moment:
//   I = 7/2, mu = +5.1487057(2) mu_N
//   N. J. Stone, IAEA table of nuclear moments (INDC(NDS)-0658).
//
// Fundamental constant:
//   mu_N/h = 7.6225932188(24) MHz/T
//   2022 CODATA recommended values (NIST).
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_SpinAPI_NuclearZeeman
#define MOD_SpinAPI_NuclearZeeman

#include "Interaction.h"
#include "ObjectParser.h"
#include "Spin.h"

#include <armadillo>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace SpinAPI
{
    struct NuclearIsotopeData
    {
        std::string canonicalName;
        std::string elementSymbol;
        unsigned int massNumber = 0;
        int twoI = 0;
        double magneticMomentNuclearMagnetons = 0.0;
    };

    class NuclearIsotopeRegistry
    {
    private:
        static std::string Normalize(std::string value)
        {
            std::string result;
            for (const char ch:value)
            {
                const unsigned char u=
                    static_cast<unsigned char>(ch);
                if (std::isalnum(u))
                    result.push_back(
                        static_cast<char>(std::toupper(u)));
            }
            return result;
        }

    public:
        static bool Lookup(
            const std::string &isotope,
            NuclearIsotopeData &data,
            std::string &error)
        {
            error.clear();
            data=NuclearIsotopeData{};

            const std::string key=Normalize(isotope);
            if (key=="51V" || key=="V51")
            {
                data.canonicalName="51V";
                data.elementSymbol="V";
                data.massNumber=51;
                data.twoI=7;
                data.magneticMomentNuclearMagnetons=
                    5.1487057;
                return true;
            }

            error =
                "unsupported nuclear isotope; authenticated registry currently contains only 51V";
            return false;
        }
    };

    class NuclearZeeman
    {
    public:
        // 2022 CODATA: mu_N/h = 7.6225932188 MHz/T.
        // Convert MHz -> cycles/ns by 1e-3, then multiply by 2*pi.
        static constexpr double
            NuclearMagnetonOverHbarRadNsPerT()
        {
            return
                6.283185307179586476925286766559 *
                7.6225932188e-3;
        }

        static double NuclearG(
            const NuclearIsotopeData &data)
        {
            if (data.twoI<=0)
                return 0.0;
            const double I=
                0.5*static_cast<double>(data.twoI);
            return
                data.magneticMomentNuclearMagnetons/I;
        }

        static double GyromagneticMagnitudeRadNsPerT(
            const NuclearIsotopeData &data)
        {
            return
                NuclearG(data) *
                NuclearMagnetonOverHbarRadNsPerT();
        }

        static double HamiltonianPrefactorRadNsPerT(
            const NuclearIsotopeData &data)
        {
            return
                -GyromagneticMagnitudeRadNsPerT(data);
        }

        static bool DataForSpin(
            const spin_ptr &spin,
            NuclearIsotopeData &data,
            std::string &error)
        {
            error.clear();
            data=NuclearIsotopeData{};

            if (spin==nullptr)
            {
                error =
                    "isotope-aware nuclear Zeeman requires a non-null spin";
                return false;
            }
            if (spin->Type()!=SpinType::Nucleus)
            {
                error =
                    "isotope-aware nuclear Zeeman requires SpinType::Nucleus";
                return false;
            }

            const auto properties=spin->Properties();
            if (properties==nullptr)
            {
                error =
                    "nuclear spin has no property parser";
                return false;
            }

            std::string isotope;
            if (!properties->Get("isotope",isotope) ||
                isotope.empty())
            {
                error =
                    "nuclear spin requires an explicit isotope property";
                return false;
            }

            if (!NuclearIsotopeRegistry::Lookup(
                    isotope,data,error))
                return false;

            if (spin->S()!=data.twoI)
            {
                error =
                    "nuclear isotope spin quantum number does not match Spin object";
                return false;
            }

            return true;
        }

        static bool CreateInteraction(
            const std::string &name,
            const spin_ptr &nucleus,
            const arma::vec &field,
            interaction_ptr &interaction,
            std::string &error)
        {
            error.clear();
            interaction.reset();

            NuclearIsotopeData data;
            if (!DataForSpin(nucleus,data,error))
                return false;

            if (field.n_elem!=3 ||
                !field.is_finite())
            {
                error =
                    "isotope-aware nuclear Zeeman field must be a finite 3-vector";
                return false;
            }

            const double prefactor=
                HamiltonianPrefactorRadNsPerT(data);
            if (!std::isfinite(prefactor))
            {
                error =
                    "isotope-aware nuclear Zeeman prefactor is non-finite";
                return false;
            }

            std::ostringstream properties;
            properties << std::setprecision(17)
                       << "type=zeeman;spins="
                       << nucleus->Name()
                       << ";field="
                       << field(0) << " "
                       << field(1) << " "
                       << field(2)
                       << ";ignoretensors=true;"
                       << "commonprefactor=false;"
                       << "prefactor=" << prefactor
                       << ";";

            auto candidate=
                std::make_shared<Interaction>(
                    name,properties.str());

            // Match the native SpinAPI interaction lifecycle: Interaction
            // objects are constructed unbound and SpinSystem owns the one
            // ParseSpinGroups/ValidateInteractions binding step.  Parsing here
            // and then adding to a SpinSystem would parse the same group twice;
            // Interaction::AddSpinList deliberately rejects duplicates.
            if (candidate==nullptr ||
                candidate->Type()!=
                    InteractionType::SingleSpin ||
                candidate->AddCommonPrefactor() ||
                !candidate->IgnoreTensors() ||
                !std::isfinite(candidate->Prefactor()))
            {
                error =
                    "failed to construct isotope-aware nuclear Zeeman interaction";
                return false;
            }

            const arma::vec candidateField=
                candidate->Field();
            const double scale=std::max({
                1.0,
                std::abs(prefactor),
                std::abs(candidate->Prefactor())
            });
            if (candidateField.n_elem!=3 ||
                !candidateField.is_finite() ||
                arma::norm(candidateField-field,2)>
                    1.0e-14*std::max(
                        1.0,arma::norm(field,2)) ||
                std::abs(
                    candidate->Prefactor()-prefactor)>
                    1.0e-14*scale)
            {
                error =
                    "failed to preserve isotope-aware nuclear Zeeman construction parameters";
                return false;
            }

            interaction=candidate;
            return true;
        }

        static bool ValidateInteraction(
            const spin_ptr &nucleus,
            const interaction_ptr &interaction,
            std::string &error,
            double relativeTolerance=1.0e-12)
        {
            error.clear();

            NuclearIsotopeData data;
            if (!DataForSpin(nucleus,data,error))
                return false;

            if (interaction==nullptr ||
                interaction->Type()!=
                    InteractionType::SingleSpin)
            {
                error =
                    "isotope-aware nuclear Zeeman must be a single-spin interaction";
                return false;
            }
            if (interaction->HasTimeDependence())
            {
                error =
                    "isotope-aware nuclear Zeeman interaction must be static";
                return false;
            }

            const auto group1=interaction->Group1();
            const auto group2=interaction->Group2();
            if (group1.size()!=1 ||
                group1.front()!=nucleus ||
                !group2.empty())
            {
                error =
                    "isotope-aware nuclear Zeeman interaction must own exactly the selected nucleus";
                return false;
            }

            if (interaction->AddCommonPrefactor())
            {
                error =
                    "isotope-aware nuclear Zeeman must set commonprefactor=false";
                return false;
            }
            if (!interaction->IgnoreTensors())
            {
                error =
                    "isotope-aware bare nuclear Zeeman must set ignoretensors=true";
                return false;
            }

            if (!std::isfinite(relativeTolerance) ||
                relativeTolerance<0.0)
            {
                error =
                    "isotope-aware nuclear Zeeman validation tolerance is invalid";
                return false;
            }

            const double expected=
                HamiltonianPrefactorRadNsPerT(data);
            const double actual=interaction->Prefactor();
            const double scale=std::max({
                1.0,std::abs(expected),std::abs(actual)});
            if (!std::isfinite(actual) ||
                std::abs(actual-expected)>
                    relativeTolerance*scale)
            {
                error =
                    "isotope-aware nuclear Zeeman prefactor does not match -g_N mu_N/hbar";
                return false;
            }

            const arma::vec field=interaction->Field();
            if (field.n_elem!=3 ||
                !field.is_finite())
            {
                error =
                    "isotope-aware nuclear Zeeman field must be a finite 3-vector";
                return false;
            }

            return true;
        }
    };
}

#endif
