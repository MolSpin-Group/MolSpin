/////////////////////////////////////////////////////////////////////////
// GeneralOrientationSampler (RunSection::General)
// ------------------
// Shared molecular-to-laboratory orientation construction for HSGeneral,
// SSGeneral and MultiSSGeneral. SpinAPI PowderGrid constructors intentionally
// expose their established solid-angle integration weights; General converts
// generated quadrature sets to normalized ensemble-average weights exactly
// once. Explicit user-supplied orientation weights are preserved verbatim.
//
// MolSpin's established powder convention maps a PowderGrid point (theta,phi)
// to active ZYZ R(0,theta,phi). The additional SO(3) sampling angle (legacy
// input name powdergamma) occupies the first ZYZ angle: R(gamma,theta,phi).
// This ordering is retained here to avoid changing established physics.
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_GeneralOrientationSampler
#define MOD_RunSection_General_GeneralOrientationSampler

#include <armadillo>
#include <iosfwd>
#include <string>
#include <vector>
#include "PowderGrid.h"

namespace RunSection::General
{
    enum class GeneralOrientationMode { Identity, Explicit, Powder2D, PowderSO3 };

    struct GeneralOrientationRequest
    {
        GeneralOrientationMode mode = GeneralOrientationMode::Identity;
        SpinAPI::PowderGridType gridType = SpinAPI::PowderGridType::Uniform;
        SpinAPI::PowderGridDomain domain = SpinAPI::PowderGridDomain::UpperHemisphere;
        int powderPoints = 1;
        int powderGridSize = 1;
        std::string powderSymmetry = "c1";
        int extraAnglePoints = 1;
        double extraAngleOffset = 0.0;
        double explicitAlpha = 0.0;
        double explicitBeta = 0.0;
        double explicitGamma = 0.0;
        double explicitWeight = 1.0;
        std::string owner = "General";
    };

    struct GeneralOrientation
    {
        double alpha = 0.0;
        double beta = 0.0;
        double gamma = 0.0;
        double weight = 1.0;
        arma::mat frameToLab = arma::eye<arma::mat>(3,3);
    };

    class GeneralOrientationSampler
    {
    public:
        static bool Build(const GeneralOrientationRequest &_request,
            std::vector<GeneralOrientation> &_out, std::ostream &_log,
            std::string &_error);
    };
}

#endif
